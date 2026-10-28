"""
PACE on LeRobot's pi0.5 (`lerobot.policies.pi05.PI05Policy`), attached from outside: no pi0.5 weight or source line
changes, and `PI05Experience.detach()` restores the original policy.

Where PACE meets pi0.5:
  - retrieval key of a call: from the SigLIP + projector image tokens that `PI05Pytorch._embed_images` returns (before
    the language model, so independent of the instruction and of the state written into the prompt). Raw key (`key`):
      grid  every camera's 16 x 16 token grid average-pooled to 4 x 4 cells, each cell unit-normalized, concatenated
            over `cameras` (default; keeps where things are);
      mean  every camera's token mean, unit-normalized, concatenated.
    The library fits a key transform on its raw keys (`core.fit_key_transform`, default PCA to 128 dims after
    centering: the raw keys share one large component, so their cosine similarities are all ~0.996 and PMA's prior
    swamps them; see pace/key_study.py); the store keeps it and every key is transformed the same way, then unit-
    normalized. The first-frame scene signature is the first camera's token grid average-pooled to `sig_grid` x
    `sig_grid`.
  - reference action chunks: the normalized (preprocessor output) chunks of `chunk_size` steps, without padding.
  - adapter: forward hooks on `model.action_in_proj` (input injection) and on the action expert's decoder layers
    `paligemma_with_expert.gemma_expert.model.layers[i]` (gated cross-attention). pi0.5 has no condition token in the
    suffix: its action tokens are the whole expert sequence, and time enters through adaRMS.
  - where in block i (the adapter's `position`), the block being x = x + g1*Attn(adaRMS1(x)); x = x + g2*MLP(adaRMS2(x)):
      post_layer  after the block: x_out + r(x_out);
      pre_mlp     between the joint attention and the MLP: with x' = x + g1*Attn(...), the block computes
                  x' + r + g2*MLP(adaRMS2(x' + r)). The MLP's input norm gets a forward pre-hook that adds r to its
                  input, and the block's forward hook adds the same r to the output (the block keeps its own copy of
                  x' as the MLP residual), so the experience enters the residual stream before the MLP reads it.
    Both are pre-norm residual branches without any norm on the residual stream itself, as in the host: a zero
    residual leaves the block bit-identical.
  - time_cond: every cross-attention also reads the block's adaRMS condition (the flow-time embedding).
  - v2 encoder features of a call: its image tokens = every camera's token grid average-pooled to 4 x 4 cells, each
    cell unit-normalized ([16 * len(cameras), D], the grid key's cells before they are flattened), and its robot
    state = the preprocessed batch's `observation.state` (normalized as pi0.5 reads it). The attachment keeps the
    state of the running `predict_action_chunk` call for its `_embed_images` hook.
  - v3 retrieval: the robot state of every call (as above) joins the alignment, and instructions are compared with
    `InstructionText` (pi0.5's own PaliGemma input token embeddings, IDF-weighted mean over the instruction's tokens,
    centered on the library's tasks), so references can come from the tasks closest to what the policy is told.

The expert's decoder layers run their own `forward` only on the expert-only path, i.e. the denoising steps of
`sample_actions` after the prefix (images + prompt) has been cached. The joint prefix+suffix training forward
(`PI05Pytorch.forward`) calls `compute_layer_complete` instead and bypasses the layer hooks, so the adapter is trained
here on the inference path: the prefix KV cache is computed without gradients and one denoising step at a sampled
flow time is scored against the flow-matching target (`flow_loss`). Since the prefix never attends to the action
tokens, that cache equals the prefix half of the joint forward.

Requires `compile_model=False` (hooks and instance wrappers are invisible to a compiled `sample_actions`) and MEM
video memory off (5-D images would be keyed by their pooled video tokens, which this module was not written for).
"""

import collections
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

from .core import (
    RETRIEVALS, SCALAR_NAMES, V3_SCALAR_NAMES, ExperienceAdapter, ExperienceRuntime, ExperienceStore,
    apply_key_transform, empty_slots, has_features, has_states, normalize_instruction, transform_id,
)

KEY_KINDS = ("grid", "mean")


def key_info(cameras: Sequence[int], sig_grid: int, key: str = "grid", transform: Optional[dict] = None) -> dict:
    """Recorded in every store, so keys computed one way are never aligned against keys computed another."""
    assert key in KEY_KINDS, f"Unknown key {key}"
    return {"host": "pi05", "key": key, "cameras": [int(c) for c in cameras], "sig_grid": int(sig_grid),
            "transform": transform_id(transform)}


def _token_grid(e: torch.Tensor, cells: int) -> torch.Tensor:
    """Image tokens [B, N, D] -> [B, cells^2, D] average-pooled grid."""
    B, N, D = e.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f"{N} image tokens are not a square grid"
    return F.adaptive_avg_pool2d(e.transpose(1, 2).reshape(B, D, side, side), cells).flatten(2).transpose(1, 2)


def raw_keys(embs: List[torch.Tensor], cameras: Sequence[int], sig_grid: int, key: str = "grid"):
    """Per-camera image tokens [B, N, D] -> raw keys [B, d] (before the key transform) and unit signatures."""
    if key == "grid":
        raw = torch.cat([F.normalize(_token_grid(embs[c].float(), 4), dim=-1).flatten(1) for c in cameras], dim=-1)
    elif key == "mean":
        raw = torch.cat([F.normalize(embs[c].float().mean(dim=1), dim=-1) for c in cameras], dim=-1)
    else:
        raise ValueError(f"Unknown key {key}")
    sig = _token_grid(embs[cameras[0]].float(), sig_grid).flatten(1)
    return raw, F.normalize(sig, dim=-1)


def image_keys(embs: List[torch.Tensor], cameras: Sequence[int], sig_grid: int, key: str = "grid",
               transform: Optional[dict] = None):
    """Per-camera image tokens [B, N, D] -> unit keys [B, d'] (transformed) and unit signatures [B, sig_grid^2 * D]."""
    raw, sig = raw_keys(embs, cameras, sig_grid, key)
    return apply_key_transform(raw, transform), sig


def vis_tokens(embs: List[torch.Tensor], cameras: Sequence[int], cells: int = 4) -> torch.Tensor:
    """Per-camera image tokens [B, N, D] -> v2 encoder image tokens [B, cells^2 * len(cameras), D] (unit cells)."""
    return torch.cat([F.normalize(_token_grid(embs[c].float(), cells), dim=-1) for c in cameras], dim=1)


def batch_state(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The preprocessed batch's robot state [B, Ds] (the last observation step if it carries several)."""
    state = batch[OBS_STATE]
    return (state[:, -1] if state.dim() == 3 else state).float()


class InstructionText:
    """
    Instruction similarity for the v3 retrieval, from pi0.5's own input token embeddings (PaliGemma's), so no extra
    model is needed: an instruction (normalized as a task key) is the IDF-weighted mean of its tokens' embeddings, IDF
    over the reference tasks (words every task shares, "put", "the", weigh little), minus the mean over the reference
    tasks (the component all instructions share), unit-normalized; the similarity of two instructions is the cosine.
    Identical instructions have similarity 1; paraphrases keep part of it through shared and related words. The fit
    is fixed once (`fit`, on the library's tasks), so later instructions never change earlier similarities.
    """

    def __init__(self, weight: torch.Tensor, tokenizer, tasks: Sequence[str] = ()) -> None:
        self.weight, self.tokenizer = weight, tokenizer  # [vocab, D] input embedding table
        self.idf: Dict[int, float] = {}
        self.default_idf = 1.0
        self.mean: Optional[torch.Tensor] = None
        self._cache: Dict[str, torch.Tensor] = {}
        if tasks:
            self.fit(tasks)

    @classmethod
    def from_policy(cls, policy, tasks: Sequence[str] = ()) -> "InstructionText":
        from transformers import AutoTokenizer

        language_model = policy.model.paligemma_with_expert.paligemma.model.language_model
        weight = language_model.get_input_embeddings().weight.detach()
        return cls(weight, AutoTokenizer.from_pretrained(policy.config.text_tokenizer_name), tasks)

    @classmethod
    def from_checkpoint(cls, policy_path: str, tokenizer_name: str = "google/paligemma-3b-pt-224",
                        tasks: Sequence[str] = ()) -> "InstructionText":
        """
        Without loading the policy: the embedding table read from the checkpoint file (CPU, ~1 GB). PaliGemma ties its
        input embeddings to its output head, and the lerobot checkpoints store only the head
        (`...paligemma.lm_head.weight`), which is then the table.
        """
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoTokenizer

        path = Path(policy_path) / "model.safetensors"
        if not path.exists():
            path = Path(hf_hub_download(policy_path, "model.safetensors"))
        with safe_open(str(path), framework="pt") as f:
            keys = [k for k in f.keys() if ".paligemma." in k and "gemma_expert" not in k]
            names = [k for k in keys if k.endswith("embed_tokens.weight")] or [k for k in keys if k.endswith("lm_head.weight")]
            assert len(names) == 1, f"{path}: input embedding / tied head keys {names}"
            weight = f.get_tensor(names[0])
        return cls(weight, AutoTokenizer.from_pretrained(tokenizer_name), tasks)

    def token_ids(self, text: str) -> List[int]:
        return list(self.tokenizer(normalize_instruction(text), add_special_tokens=False)["input_ids"])

    def fit(self, tasks: Sequence[str]) -> None:
        tasks = list(dict.fromkeys(normalize_instruction(t) for t in tasks))
        df = collections.Counter()
        for task in tasks:
            df.update(set(self.token_ids(task)))
        n = len(tasks)
        self.idf = {tok: math.log((1 + n) / (1 + c)) + 1.0 for tok, c in df.items()}
        self.default_idf = math.log(1 + n) + 1.0  # a word no reference task uses
        self.mean, self._cache = None, {}
        self.mean = torch.stack([self._raw(t) for t in tasks]).mean(dim=0)

    def _raw(self, text: str) -> torch.Tensor:
        ids = self.token_ids(text)
        if not ids:
            return torch.zeros(self.weight.shape[1])
        w = torch.tensor([self.idf.get(i, self.default_idf) for i in ids])
        e = self.weight[torch.tensor(ids, device=self.weight.device)].float().cpu()
        return (w[:, None] * e).sum(dim=0) / w.sum()

    def embed(self, text: str) -> torch.Tensor:
        key = normalize_instruction(text)
        if key not in self._cache:
            v = self._raw(key)
            self._cache[key] = F.normalize(v - self.mean if self.mean is not None else v, dim=0)
        return self._cache[key]

    def similarities(self, query: str, tasks: Sequence[str]) -> List[float]:
        q = self.embed(query)
        return [float(self.embed(t) @ q) for t in tasks]


def policy_dims(policy) -> dict:
    """Shapes the adapter must match."""
    model = policy.model
    expert = model.paligemma_with_expert.gemma_expert.model
    return {
        "hidden_size": model.action_in_proj.out_features,
        "depth": len(expert.layers),
        "num_heads": expert.config.num_attention_heads,
        "chunk_len": policy.config.chunk_size,
        "action_dim": policy.config.output_features[ACTION].shape[0],
    }


def resolve_layers(spec, depth: int) -> List[int]:
    """
    Injection layers from a spec: layer indices and/or presets `all`, `every<N>` (layers N-1, 2N-1, ...),
    `last<N>`, `first<N>`, `last_half`, `first_half`; e.g. ["every2"], ["last4"], ["0", "8", "17"].
    """
    if isinstance(spec, (str, int)):
        spec = [spec]
    layers = set()
    for item in spec:
        item = str(item).strip().lower()
        if item == "all":
            layers.update(range(depth))
        elif item == "last_half":
            layers.update(range(depth // 2, depth))
        elif item == "first_half":
            layers.update(range(depth // 2))
        elif item.startswith("every"):
            n = int(item[5:])
            layers.update(i for i in range(depth) if (i + 1) % n == 0)
        elif item.startswith("last"):
            layers.update(range(max(depth - int(item[4:]), 0), depth))
        elif item.startswith("first"):
            layers.update(range(min(int(item[5:]), depth)))
        else:
            index = int(item)
            layers.add(index if index >= 0 else depth + index)
    out = sorted(layers)
    assert out and all(0 <= i < depth for i in out), f"injection layers {spec} -> {out} for depth {depth}"
    return out


def build_adapter(policy, inject_layers=("every2",), max_refs: int = 8, input_inject: bool = True,
                  position: str = "post_layer", time_cond: bool = False, num_heads: Optional[int] = None,
                  encoder: str = "v1", feature_shapes=None, enc_dim: int = 512, enc_layers: int = 2,
                  enc_heads: int = 8, act_segments: int = 10, retrieval: str = "classic", generic: str = "none",
                  generic_dim: int = 256, generic_time: bool = True,
                  generic_cond_rank: int = 64, relevance: str = "none",
                  ref_isolation: bool = False) -> ExperienceAdapter:
    """
    `feature_shapes` = ((V, Dv), Ds) of the library's image tokens and states (required by the v2 encoder);
    `retrieval` fixes the per-reference scalars the adapter reads (classic: SCALAR_NAMES, v3: V3_SCALAR_NAMES);
    `generic` "mlp" adds the v4 reference-independent branch (`core.GenericResidual`); `relevance` "gate" the v6
    per-reference relevance gate (the reference branch can decline its references); `ref_isolation` (v8) keeps the
    current call out of the reference tokens (`core.ExperienceAdapter`).
    """
    assert retrieval in RETRIEVALS, f"Unknown retrieval {retrieval}"
    dims = policy_dims(policy)
    v2 = {}
    if encoder == "v2":
        assert feature_shapes is not None, "the v2 encoder needs the library's feature shapes (built with --features 1)"
        (V, Dv), Ds = feature_shapes
        v2 = dict(vis_tokens=int(V), vis_dim=int(Dv), state_dim=int(Ds), enc_dim=enc_dim, enc_layers=enc_layers,
                  enc_heads=enc_heads, act_segments=act_segments)
    return ExperienceAdapter(
        hidden_size=dims["hidden_size"], num_heads=num_heads or dims["num_heads"], depth=dims["depth"],
        inject_layers=resolve_layers(inject_layers, dims["depth"]), action_dim=dims["action_dim"],
        chunk_len=dims["chunk_len"], max_refs=max_refs, input_inject=input_inject, position=position,
        time_cond=time_cond, cond_dim=dims["hidden_size"], encoder=encoder,
        scalar_names=V3_SCALAR_NAMES if retrieval == "v3" else SCALAR_NAMES, generic=generic, generic_dim=generic_dim,
        generic_time=generic_time, generic_cond_rank=generic_cond_rank, relevance=relevance,
        ref_isolation=ref_isolation, **v2,
    )


class PI05Experience:
    """
    PACE attached to a PI05Policy. Two ways to feed the adapter:
      - explicit (training, offline scoring): `set_exp(exp)` before the denoising steps, `set_exp(None)` after;
      - episodic (evaluation): `begin(task_keys)` before a batch of episodes, after which every policy call computes
        its keys, advances one `ExperienceRuntime` per batch row and feeds the references; `end(...)` closes the
        episodes, returns their info and (optionally) turns them into store entries.
    Without references (no adapter inputs, or no row with a valid reference) nothing is added and the policy output
    is bit-identical to the base policy, unless the adapter has a v4 generic branch: that one acts on every call while
    `generic_on` (default True for such an adapter; False gives the base policy again).
    """

    def __init__(
        self,
        policy,
        adapter: ExperienceAdapter,
        store: Optional[ExperienceStore] = None,
        cameras: Optional[Sequence[int]] = None,
        sig_grid: Optional[int] = None,
        key: Optional[str] = None,
        key_transform: Optional[dict] = None,
        runtime_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Keys follow the store (its info and fitted transform) when one is given; `cameras` / `sig_grid` / `key` /
        `key_transform` must then agree with it. Without a store: (0, 1), 4, "grid" and no transform by default.
        """
        assert not policy.config.compile_model, "PACE needs compile_model=False"
        assert not (policy.config.use_visual_memory or policy.config.use_proprioceptive_memory), \
            "PACE was not written for MEM memory inputs"
        self.policy, self.model = policy, policy.model
        dims = policy_dims(policy)
        for name in ("hidden_size", "depth", "chunk_len", "action_dim"):
            assert adapter.config[name] == dims[name], f"adapter {name}={adapter.config[name]} vs policy {dims[name]}"
        self.device = next(policy.parameters()).device
        self.adapter = adapter.to(self.device)
        info = store.info if store is not None and store.info else {}
        if info and key_transform is None:
            key_transform = store.key_transform
        self.cameras = tuple(int(c) for c in (cameras if cameras is not None else info.get("cameras", (0, 1))))
        self.sig_grid = int(sig_grid if sig_grid is not None else info.get("sig_grid", 4))
        self.key = key if key is not None else info.get("key", "grid")
        self.key_transform = key_transform
        self.key_info = key_info(self.cameras, self.sig_grid, self.key, self.key_transform)
        self.store = store
        if info and info != self.key_info:
            raise ValueError(f"store keys {info} differ from this attachment's {self.key_info}")
        self.runtime_kwargs = dict(runtime_kwargs or {})
        self.chunk_shape = (dims["chunk_len"], dims["action_dim"])
        self.position = adapter.config.get("position", "post_layer")
        assert self.position in ExperienceAdapter.POSITIONS, f"Unknown position {self.position}"
        # v2 encoder: every call also feeds image tokens and robot states, so the store must hold them
        self.features = adapter.uses_features
        if self.features:
            assert len(self.cameras) * 16 == adapter.config["vis_tokens"], \
                f"{len(self.cameras)} cameras x 16 cells vs the adapter's {adapter.config['vis_tokens']} image tokens"
            if store is not None:
                lacking = [ep["name"] for ep in store.episodes if not has_features(ep)]
                if lacking:
                    raise ValueError(f"the v2 encoder reads image tokens / states, which {len(lacking)} store episodes "
                                     f"(e.g. {lacking[0]}) lack: build the library with --features 1")
        # v3 retrieval: robot states join the alignment, instructions are compared, the adapter reads 3 more scalars
        self.retrieval = self.runtime_kwargs.get("retrieval", "classic")
        assert self.retrieval in RETRIEVALS, f"Unknown retrieval {self.retrieval}"
        wanted = V3_SCALAR_NAMES if self.retrieval == "v3" else SCALAR_NAMES
        if tuple(adapter.scalar_names) != wanted:
            raise ValueError(f"the adapter reads the scalars {adapter.scalar_names}, the {self.retrieval} retrieval "
                             f"gives {wanted}: use the retrieval it was trained with")
        self.n_scalars = len(wanted)
        self.runtime_kwargs["n_scalars"] = self.n_scalars
        self.needs_state = self.features or self.retrieval == "v3"
        self.text: Optional[InstructionText] = None
        if self.retrieval == "v3":
            if store is not None:
                lacking = [ep["name"] for ep in store.episodes if not has_states(ep)]
                if lacking:
                    raise ValueError(f"the v3 retrieval aligns on robot states, which {len(lacking)} store episodes "
                                     f"(e.g. {lacking[0]}) lack: rebuild the library with this build_library")
            # fitted once on the demos' tasks (as in training; instructions of accumulated rollouts, also those a
            # resumed run reloads, are compared but never fitted on)
            demo_tasks = sorted({ep["task_key"] for ep in store.episodes if ep["source"] == "demo"})                 if store is not None else []
            self.text = InstructionText.from_policy(policy, demo_tasks)
            self.runtime_kwargs.setdefault("task_text", self.text.similarities)
        # v4: the generic branch acts on every call while attached (with or without references); generic_on=False
        # switches it off (the host alone without references, e.g. a base-policy comparison)
        self.generic_on = adapter.has_generic
        self._state: Optional[torch.Tensor] = None  # robot state of the running predict_action_chunk call
        self.encoded = None
        self.runtimes: Optional[List[ExperienceRuntime]] = None
        self._inside: Optional[int] = None  # injection layer whose forward is running (pre_mlp bookkeeping)
        self._pending: Optional[torch.Tensor] = None  # residual added at its MLP input, to add to its output too
        self.handles = []
        self.attached = False
        self.attach()

    # --- hooks ---
    def attach(self) -> None:
        if self.attached:
            return
        expert_layers = self.model.paligemma_with_expert.gemma_expert.model.layers
        self.handles.append(self.model.action_in_proj.register_forward_hook(self._input_hook))
        for i in self.adapter.inject_layers:
            layer = expert_layers[i]
            self.handles.append(layer.register_forward_pre_hook(self._enter_hook(i)))
            self.handles.append(layer.register_forward_hook(self._layer_hook(i), with_kwargs=True))
            if self.position == "pre_mlp":
                self.handles.append(
                    layer.post_attention_layernorm.register_forward_pre_hook(self._pre_mlp_hook(i), with_kwargs=True)
                )
        original_embed = self.model._embed_images  # bound class method
        original_predict = self.policy.predict_action_chunk

        def embed_images(images, img_masks):
            embs = original_embed(images, img_masks)
            if self.runtimes is not None:
                self._on_call(embs)
            return embs

        def predict_action_chunk(batch, **kwargs):
            if self.runtimes is not None and self.needs_state:
                self._state = batch_state(batch)
            try:
                actions = original_predict(batch, **kwargs)
            finally:
                self._state = None
                if self.runtimes is not None:
                    self.encoded = None
            if self.runtimes is not None:
                chunks = actions.float().cpu().numpy()
                for runtime, chunk in zip(self.runtimes, chunks):
                    runtime.record_prediction(chunk)
            return actions

        self.model._embed_images = embed_images
        self.policy.predict_action_chunk = predict_action_chunk
        self.attached = True

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for owner, name in ((self.model, "_embed_images"), (self.policy, "predict_action_chunk")):
            if name in vars(owner):
                delattr(owner, name)
        self.encoded, self.runtimes, self.attached = None, None, False
        self._inside, self._pending = None, None

    @property
    def acting(self) -> bool:
        """Whether the adapter adds anything to the next denoising steps (references fed, or the generic branch on)."""
        return self.encoded is not None or self.generic_on

    def _input_hook(self, module, args, output):
        if not self.acting:
            return None
        residual = self.adapter.input_residual(self.encoded, output.shape[1], generic=self.generic_on)
        if residual is None:
            return None
        assert residual.shape[1] == output.shape[1] and residual.shape[0] in (1, output.shape[0]),             f"action tokens {tuple(output.shape)} vs {tuple(residual.shape)}"
        return output + residual.to(output.dtype)

    def _enter_hook(self, index: int):
        def hook(module, args):
            self._inside, self._pending = index, None

        return hook

    def _pre_mlp_hook(self, index: int):
        """On the MLP's input norm of block `index`: add r to the norm input and keep r for the block output."""

        def hook(module, args, kwargs):
            # Only inside this block's own forward (the joint training path calls the norm without the block)
            if not self.acting or self._inside != index:
                return None
            hidden = args[0]
            residual = self.adapter.layer_residual(index, hidden, self.encoded, cond=kwargs.get("cond"),
                                                   generic=self.generic_on)
            if residual is None:
                return None
            residual = residual.to(hidden.dtype)
            self._pending = residual
            return (hidden + residual, *args[1:]), kwargs

        return hook

    def _layer_hook(self, index: int):
        def hook(module, args, kwargs, output):
            pending = self._pending
            self._inside, self._pending = None, None
            if not self.acting:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            if self.position == "pre_mlp":
                residual = pending
                assert residual is not None, f"expert layer {index}: the MLP input norm was not called"
            else:
                residual = self.adapter.layer_residual(index, hidden, self.encoded, cond=kwargs.get("adarms_cond"),
                                                       generic=self.generic_on)
                if residual is None:
                    return None
                residual = residual.to(hidden.dtype)
            hidden = hidden + residual
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

        return hook

    # --- explicit adapter inputs ---
    def set_exp(self, exp: Optional[Dict[str, torch.Tensor]]) -> None:
        """Adapter inputs for the next denoising steps ([B, max_refs, ...] tensors), or None for the base policy."""
        if exp is None or not bool(exp["valid"].any()):
            self.encoded = None
            return
        exp = {k: v.to(self.device) for k, v in exp.items()}
        self.encoded = self.adapter.encode(exp)

    # --- episodic use ---
    def begin(self, task_keys: Sequence[str], condition: str = "real", seeds: Optional[Sequence[int]] = None,
              true_tasks: Optional[Sequence[str]] = None, query_texts: Optional[Sequence[Optional[str]]] = None) -> None:
        """
        Start one episode per batch row (task keys normalized with `core.normalize_instruction`; the stored rollout
        keeps its row's key). v3 retrieval: `query_texts` (default: the task keys; None entries = no instruction) are
        compared with the stored tasks' instructions, and `true_tasks` (default: the task keys) are the envs' tasks,
        which the other_task control never draws from.
        """
        assert self.store is not None or condition == "none", "a store is needed for references"
        store = self.store if self.store is not None else ExperienceStore(info=self.key_info,
                                                                          key_transform=self.key_transform)
        seeds = list(seeds) if seeds is not None else list(range(len(task_keys)))
        true_tasks = list(true_tasks) if true_tasks is not None else list(task_keys)
        query_texts = list(query_texts) if query_texts is not None else list(task_keys)
        kwargs = dict(self.runtime_kwargs)
        kwargs.setdefault("max_refs", self.adapter.config["max_refs"])
        self.runtimes = [
            ExperienceRuntime(store, task_key, condition=condition, chunk_shape=self.chunk_shape, seed=int(seed),
                              device=self.device, features=self.features, true_task=true_task, query_text=text,
                              **kwargs)
            for task_key, seed, true_task, text in zip(task_keys, seeds, true_tasks, query_texts)
        ]
        self.encoded = None

    def _on_call(self, embs: List[torch.Tensor]) -> None:
        keys, sigs = image_keys(embs, self.cameras, self.sig_grid, self.key, self.key_transform)
        assert keys.shape[0] == len(self.runtimes), f"batch of {keys.shape[0]} vs {len(self.runtimes)} episodes"
        shapes = None
        if self.needs_state:
            assert self._state is not None, "no robot state: _embed_images ran outside predict_action_chunk"
        if self.features:
            vis, states = vis_tokens(embs, self.cameras), self._state
            assert states.shape[-1] == self.adapter.config["state_dim"], \
                f"robot state of {states.shape[-1]} dims vs the adapter's {self.adapter.config['state_dim']}"
            shapes = (tuple(vis.shape[1:]), int(states.shape[-1]))
            rows = [runtime(key, sig, v, s) for runtime, key, sig, v, s in zip(self.runtimes, keys, sigs, vis, states)]
        elif self.needs_state:
            rows = [runtime(key, sig, None, s) for runtime, key, sig, s in zip(self.runtimes, keys, sigs, self._state)]
        else:
            rows = [runtime(key, sig) for runtime, key, sig in zip(self.runtimes, keys, sigs)]
        if all(row is None for row in rows):
            self.encoded = None
            return
        max_refs = self.adapter.config["max_refs"]
        rows = [row if row is not None else empty_slots(1, max_refs, self.chunk_shape, shapes, n_scalars=self.n_scalars)
                for row in rows]
        self.set_exp({k: torch.cat([row[k] for row in rows]) for k in rows[0]})

    def end(self, successes: Sequence[bool], num_calls: Optional[Sequence[int]] = None, add_to_store: bool = False,
            names: Optional[Sequence[str]] = None, metas: Optional[Sequence[dict]] = None) -> List[dict]:
        """
        Close the batch of episodes: per row its runtime info and, with `add_to_store`, its rollout (cut to its first
        `num_calls[b]` calls) added to the store. Returns the infos.
        """
        assert self.runtimes is not None, "end() without begin()"
        infos = []
        for b, runtime in enumerate(self.runtimes):
            info = runtime.info()
            if add_to_store:
                episode = runtime.to_episode(
                    bool(successes[b]), self.policy.config.n_action_steps,
                    num_calls=None if num_calls is None else int(num_calls[b]),
                    name=None if names is None else names[b], meta=None if metas is None else metas[b],
                )
                if episode is not None:
                    if self.store is None:
                        self.store = ExperienceStore(info=self.key_info, key_transform=self.key_transform)
                    self.store.add(episode)
                    info["stored_as"] = episode["name"]
            infos.append(info)
        self.runtimes, self.encoded = None, None
        return infos


# === Training path (pi0.5 frozen, adapter trainable) ===
@torch.no_grad()
def prefix_cache(policy, batch: Dict[str, torch.Tensor]):
    """Prefix (images + prompt) KV cache of a preprocessed batch, as `PI05Pytorch.sample_actions` builds it."""
    model = policy.model
    images, img_masks = policy._preprocess_images(batch)
    states, state_masks = policy._prepare_memory_states(batch)
    tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks, states, state_masks
    )
    att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    mask_dtype = prefix_embs.dtype if model.use_typed_attention_masks else None
    att_4d = model._prepare_attention_masks_4d(att_2d, dtype=mask_dtype)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=att_4d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values


def flow_loss(policy, prefix, actions: torch.Tensor, noise: Optional[torch.Tensor] = None,
              time: Optional[torch.Tensor] = None, per_step: bool = False) -> torch.Tensor:
    """
    Per-sample flow-matching loss of one denoising step on the inference path (the adapter, if fed with `set_exp`,
    acts through the hooks). actions: [B, chunk, A] normalized, unpadded. Returns [B] ([B, chunk] with `per_step`).
    """
    model = policy.model
    padded = F.pad(actions, (0, policy.config.max_action_dim - actions.shape[-1]))
    if noise is None:
        noise = model.sample_noise(padded.shape, padded.device)
    if time is None:
        time = model.sample_time(padded.shape[0], padded.device)
    x_t, u_t, model_time = make_flow_matching_inputs(padded, noise, time, convention=FlowConvention.NOISE_AT_ONE)
    prefix_pad_masks, past_key_values = prefix
    v_t = model.denoise_step(prefix_pad_masks=prefix_pad_masks, past_key_values=past_key_values, x_t=x_t,
                             timestep=model_time)
    A = actions.shape[-1]
    err = ((v_t.float() - u_t.float())[..., :A] ** 2).mean(dim=2)
    return err if per_step else err.mean(dim=1)


@torch.no_grad()
def call_keys(policy, batch: Dict[str, torch.Tensor], cameras: Sequence[int], sig_grid: int, key: str = "grid",
              features: bool = False):
    """
    Raw retrieval keys (before the key transform) and signatures of a preprocessed batch (no language model); with
    `features` also the v2 encoder's image tokens [B, V, D] and robot states [B, Ds].
    """
    images, img_masks = policy._preprocess_images(batch)
    embs = policy.model._embed_images(images, img_masks)
    raw, sig = raw_keys(embs, cameras, sig_grid, key)
    if not features:
        return raw, sig
    return raw, sig, vis_tokens(embs, cameras), batch_state(batch)
