"""
PACE on LeRobot's pi0.5 (`lerobot.policies.pi05.PI05Policy`), attached from outside: no pi0.5 weight or source line
changes, and `PI05Experience.detach()` restores the original policy.

Where PACE meets pi0.5:
  - retrieval key of a call: the SigLIP + projector image tokens that `PI05Pytorch._embed_images` returns (before the
    language model, so independent of the instruction and of the state written into the prompt), mean-pooled per
    camera, unit-normalized, concatenated over `cameras` and normalized again. The first-frame scene signature is the
    first camera's token grid average-pooled to `sig_grid` x `sig_grid`.
  - reference action chunks: the normalized (preprocessor output) chunks of `chunk_size` steps, without padding.
  - adapter: forward hooks on `model.action_in_proj` (input injection) and on the action expert's decoder layers
    `paligemma_with_expert.gemma_expert.model.layers[i]` (gated cross-attention after block i). pi0.5 has no
    condition token in the suffix: its action tokens are the whole expert sequence, and time enters through adaRMS.

The expert's decoder layers run their own `forward` only on the expert-only path, i.e. the denoising steps of
`sample_actions` after the prefix (images + prompt) has been cached. The joint prefix+suffix training forward
(`PI05Pytorch.forward`) calls `compute_layer_complete` instead and bypasses the layer hooks, so the adapter is trained
here on the inference path: the prefix KV cache is computed without gradients and one denoising step at a sampled
flow time is scored against the flow-matching target (`flow_loss`). Since the prefix never attends to the action
tokens, that cache equals the prefix half of the joint forward.

Requires `compile_model=False` (hooks and instance wrappers are invisible to a compiled `sample_actions`) and MEM
video memory off (5-D images would be keyed by their pooled video tokens, which this module was not written for).
"""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
from lerobot.policies.common.vla_utils import make_att_2d_masks
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .core import ExperienceAdapter, ExperienceRuntime, ExperienceStore, empty_slots


def key_info(cameras: Sequence[int], sig_grid: int) -> dict:
    """Recorded in every store, so keys computed one way are never aligned against keys computed another."""
    return {"host": "pi05", "key": "image_token_mean", "cameras": [int(c) for c in cameras], "sig_grid": int(sig_grid)}


def image_keys(embs: List[torch.Tensor], cameras: Sequence[int], sig_grid: int):
    """Per-camera image tokens [B, N, D] -> keys [B, len(cameras) * D] and signatures [B, sig_grid^2 * D], unit."""
    keys = torch.cat([F.normalize(embs[c].float().mean(dim=1), dim=-1) for c in cameras], dim=-1)
    keys = F.normalize(keys, dim=-1)
    first = embs[cameras[0]].float()
    B, N, D = first.shape
    side = int(round(N ** 0.5))
    assert side * side == N, f"{N} image tokens are not a square grid"
    grid = first.transpose(1, 2).reshape(B, D, side, side)
    sig = F.adaptive_avg_pool2d(grid, sig_grid).flatten(1)
    return keys, F.normalize(sig, dim=-1)


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


def build_adapter(policy, inject_every: int = 2, inject_layers: Optional[Sequence[int]] = None, max_refs: int = 8,
                  input_inject: bool = True, num_heads: Optional[int] = None) -> ExperienceAdapter:
    dims = policy_dims(policy)
    return ExperienceAdapter(
        hidden_size=dims["hidden_size"], num_heads=num_heads or dims["num_heads"], depth=dims["depth"],
        inject_every=inject_every, inject_layers=inject_layers, action_dim=dims["action_dim"],
        chunk_len=dims["chunk_len"], max_refs=max_refs, input_inject=input_inject,
    )


class PI05Experience:
    """
    PACE attached to a PI05Policy. Two ways to feed the adapter:
      - explicit (training, offline scoring): `set_exp(exp)` before the denoising steps, `set_exp(None)` after;
      - episodic (evaluation): `begin(task_keys)` before a batch of episodes, after which every policy call computes
        its keys, advances one `ExperienceRuntime` per batch row and feeds the references; `end(...)` closes the
        episodes, returns their info and (optionally) turns them into store entries.
    Without references (no adapter inputs, or no row with a valid reference) nothing is added and the policy output
    is bit-identical to the base policy.
    """

    def __init__(
        self,
        policy,
        adapter: ExperienceAdapter,
        store: Optional[ExperienceStore] = None,
        cameras: Sequence[int] = (0, 1),
        sig_grid: int = 4,
        runtime_kwargs: Optional[dict] = None,
    ) -> None:
        assert not policy.config.compile_model, "PACE needs compile_model=False"
        assert not (policy.config.use_visual_memory or policy.config.use_proprioceptive_memory), \
            "PACE was not written for MEM memory inputs"
        self.policy, self.model = policy, policy.model
        dims = policy_dims(policy)
        for name in ("hidden_size", "depth", "chunk_len", "action_dim"):
            assert adapter.config[name] == dims[name], f"adapter {name}={adapter.config[name]} vs policy {dims[name]}"
        self.device = next(policy.parameters()).device
        self.adapter = adapter.to(self.device)
        self.cameras, self.sig_grid = tuple(int(c) for c in cameras), int(sig_grid)
        self.key_info = key_info(self.cameras, self.sig_grid)
        self.store = store
        if store is not None and store.info and store.info != self.key_info:
            raise ValueError(f"store keys {store.info} differ from this attachment's {self.key_info}")
        self.runtime_kwargs = dict(runtime_kwargs or {})
        self.chunk_shape = (dims["chunk_len"], dims["action_dim"])
        self.encoded = None
        self.runtimes: Optional[List[ExperienceRuntime]] = None
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
            self.handles.append(expert_layers[i].register_forward_hook(self._layer_hook(i)))
        original_embed = self.model._embed_images  # bound class method
        original_predict = self.policy.predict_action_chunk

        def embed_images(images, img_masks):
            embs = original_embed(images, img_masks)
            if self.runtimes is not None:
                self._on_call(embs)
            return embs

        def predict_action_chunk(batch, **kwargs):
            try:
                actions = original_predict(batch, **kwargs)
            finally:
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

    def _input_hook(self, module, args, output):
        if self.encoded is None or self.encoded[3] is None:
            return None
        residual = self.encoded[3]
        assert output.shape[:2] == residual.shape[:2], f"action tokens {tuple(output.shape)} vs {tuple(residual.shape)}"
        return output + residual.to(output.dtype)

    def _layer_hook(self, index: int):
        def hook(module, args, output):
            if self.encoded is None:
                return None
            hidden = output[0] if isinstance(output, tuple) else output
            residual = self.adapter.layer_residual(index, hidden, self.encoded)
            if residual is None:
                return None
            hidden = hidden + residual.to(hidden.dtype)
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
    def begin(self, task_keys: Sequence[str], condition: str = "real", seeds: Optional[Sequence[int]] = None) -> None:
        """Start one episode per batch row (task keys normalized with `core.normalize_instruction`)."""
        assert self.store is not None or condition == "none", "a store is needed for references"
        store = self.store if self.store is not None else ExperienceStore(info=self.key_info)
        seeds = list(seeds) if seeds is not None else list(range(len(task_keys)))
        kwargs = dict(self.runtime_kwargs)
        kwargs.setdefault("max_refs", self.adapter.config["max_refs"])
        self.runtimes = [
            ExperienceRuntime(store, task_key, condition=condition, chunk_shape=self.chunk_shape, seed=int(seed),
                              device=self.device, **kwargs)
            for task_key, seed in zip(task_keys, seeds)
        ]
        self.encoded = None

    def _on_call(self, embs: List[torch.Tensor]) -> None:
        keys, sigs = image_keys(embs, self.cameras, self.sig_grid)
        assert keys.shape[0] == len(self.runtimes), f"batch of {keys.shape[0]} vs {len(self.runtimes)} episodes"
        rows = [runtime(key, sig) for runtime, key, sig in zip(self.runtimes, keys, sigs)]
        if all(row is None for row in rows):
            self.encoded = None
            return
        max_refs = self.adapter.config["max_refs"]
        rows = [row if row is not None else empty_slots(1, max_refs, self.chunk_shape) for row in rows]
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
                        self.store = ExperienceStore(info=self.key_info)
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
              time: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Per-sample flow-matching loss of one denoising step on the inference path (the adapter, if fed with `set_exp`,
    acts through the hooks). actions: [B, chunk, A] normalized, unpadded. Returns [B].
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
    return ((v_t.float() - u_t.float())[..., :A] ** 2).mean(dim=(1, 2))


@torch.no_grad()
def call_keys(policy, batch: Dict[str, torch.Tensor], cameras: Sequence[int], sig_grid: int):
    """Retrieval keys and signatures of a preprocessed batch of observations (no language model pass)."""
    images, img_masks = policy._preprocess_images(batch)
    embs = policy.model._embed_images(images, img_masks)
    return image_keys(embs, cameras, sig_grid)
