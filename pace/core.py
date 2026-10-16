"""
PACE core: Progress-Aligned Cross-episode Experience, independent of the host policy.

Every stored episode keeps, per policy call, a retrieval key (a unit vector computed by the host policy from the
call's observation) and the normalized action chunk executed from that call on, plus its outcome and a first-frame
scene signature. The current episode is aligned to every candidate stored episode of its task with Progressive Memory
Alignment (PMA, from RTCF, arXiv 2608.04527): a causal subsequence-DTW frontier over the whole execution history, so
frames of different stages that look alike are told apart by what came before. At every call the best-aligned
episodes are the references. What a reference did from its aligned position (its action chunk, one token per step),
where it is (progress, similarity, mean alignment cost, call indices) and how it ended (success) become reference
tokens (encoder v2 also reads the reference's and the current call's image tokens and robot states, and lets the
references attend to each other; see `ExperienceAdapter`). The host's action network reads them through
cross-attention layers inserted after its blocks, whose output
projection is zero-initialized, and the score-weighted reference chunk is added, step by step, to the action tokens'
input through a zero-initialized projection. A sample without references gets an exactly zero residual, so the host
policy is unchanged until the adapter is trained and given references. The v3 retrieval (`ExperienceRuntime`,
`RETRIEVALS`) also aligns on robot states and draws candidates from the tasks closest in instruction and scene.

Ported from MemoryVLA's `vla/experience_retrieval.py` (experience-retrieval branch, v2 adapter). The host-specific
parts (where the keys come from, where the residuals are added) live in `pace/pi05.py`.
"""

import collections
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Call indices are divided by this before entering the adapter (LIBERO episodes are at most ~52 calls of 10 steps)
CALL_SCALE = 64.0
SCALAR_NAMES = ("ref_progress", "similarity", "align_cost", "call", "ref_call", "ref_length")
# v3 retrieval also tells the adapter how sure the retrieval is: the instruction similarity of the reference's task
# (0 when no instruction was used), whether an instruction was used, and the robot-state cost at the matched step
V3_SCALAR_NAMES = SCALAR_NAMES + ("task_similarity", "text_known", "state_cost")
CONDITIONS = ("none", "real", "unaligned", "other_task")
# classic: the references are the task's own episodes (exact instruction match), aligned on image keys (v1 and v2);
# v3: robot states join the alignment cost and the candidates come from the tasks closest in instruction and scene
RETRIEVALS = ("classic", "v3")


def normalize_instruction(text: str) -> str:
    """Task key of an instruction: lowercase, typographic apostrophes folded, whitespace collapsed."""
    return " ".join(text.replace("’", "'").replace("_", " ").strip().lower().split())


def reference_scalars(j: int, similarity: float, align_cost: float, call: int, ref_length: int) -> List[float]:
    """Scalar features of a reference at its step `j` while the current episode is at call `call` (SCALAR_NAMES)."""
    return [j / max(ref_length - 1, 1), similarity, align_cost, call / CALL_SCALE, j / CALL_SCALE,
            ref_length / CALL_SCALE]


def state_cost(a: torch.Tensor, b: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Alignment cost of normalized robot states (broadcast over the leading dims): 1 - exp(-mean squared difference /
    sigma^2), in [0, 1) like the image cost 1 - cos. Robot states do not change with the camera, the lighting or
    image noise, which move the image keys.
    """
    return 1.0 - torch.exp(-((a.float() - b.float()) ** 2).mean(dim=-1) / sigma ** 2)


def choose_tasks(text_sims: Optional[torch.Tensor], scene_sims: Optional[torch.Tensor], text_tasks: int,
                 scene_tasks: int) -> List[int]:
    """
    v3 candidate tasks (indices into the K tasks the similarities are over): the `text_tasks` tasks whose instruction
    is closest to the current one, plus the `scene_tasks` (`text_tasks + scene_tasks` without an instruction) whose
    episodes' first frames are closest to the current first frame. Tasks with a -inf scene similarity (no episode
    to read) are never chosen for their scene.
    """
    chosen: List[int] = []
    n_scene = scene_tasks
    if text_sims is not None:
        chosen = torch.argsort(text_sims.float(), descending=True)[:text_tasks].tolist()
    else:
        n_scene = text_tasks + scene_tasks
    if scene_sims is not None:
        added = 0
        for k in torch.argsort(scene_sims.float(), descending=True).tolist():
            if added >= n_scene or not math.isfinite(float(scene_sims[k])):
                break
            if k not in chosen:
                chosen.append(k)
                added += 1
    return chosen


# === Alignment: Progressive Memory Alignment (PMA) ===
def pad_keys(keys: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """[L_p, D] key sequences -> zero-padded [P, L_max, D] and lengths [P]."""
    lengths = torch.tensor([k.shape[0] for k in keys])
    out = keys[0].new_zeros(len(keys), int(lengths.max()), keys[0].shape[-1])
    for p, k in enumerate(keys):
        out[p, : k.shape[0]] = k
    return out, lengths


class ProgressiveAlignment:
    """
    PMA (RTCF, arXiv 2608.04527): causal subsequence alignment of the running episode to P stored episodes at once.

    For every memory p and position q, J(p, q) is the lowest cumulative cost of aligning all calls so far to memory p
    with the latest call at q. The local cost is 1 - cos(call key, memory key); each call advances the episode by one
    step while the memory position stays or advances by at most `vmax`, with a soft prior gamma * |advance - 1| that
    favors unit progress; the first call may match any position (open begin). After each call, the endpoint
    argmin_q J(p, q) is the aligned position in memory p and J at the endpoint divided by the number of calls its mean
    cost, which ranks the memories. Every position keeps its best path rather than one committed estimate, so later
    calls can overturn an early match between stages that look alike. Only the latest frontier is stored.

    With the memories' robot states ([P, L, Ds], v3 retrieval) the local cost becomes
    (1 - state_weight) * (1 - cos) + state_weight * state_cost (`state_cost`), and `self.state_cost` holds the state
    cost at the aligned positions after every step; without them everything is as above.
    """

    def __init__(self, keys: torch.Tensor, lengths: torch.Tensor, vmax: int = 3, gamma: float = 0.01,
                 states: Optional[torch.Tensor] = None, state_weight: float = 0.0, state_sigma: float = 0.3) -> None:
        self.keys = keys.float()  # [P, L, D] unit vectors, zero-padded
        positions = torch.arange(keys.shape[1], device=keys.device)
        self.valid = positions[None] < lengths.to(keys.device)[:, None]
        self.vmax, self.gamma = vmax, gamma
        self.states = None if states is None else states.float().to(keys.device)  # [P, L, Ds], zero-padded
        self.state_weight, self.state_sigma = float(state_weight), float(state_sigma)
        self.state_cost: Optional[torch.Tensor] = None
        self.J: Optional[torch.Tensor] = None
        self.calls = 0

    def local_cost(self, query: torch.Tensor, query_state: Optional[torch.Tensor]):
        """Image similarities [P, L] and local costs [P, L] (+inf past each memory's end), state costs [P, L] or None."""
        sims = torch.matmul(self.keys, query.float().to(self.keys.device)).clamp(-1.0, 1.0)  # [P, L]
        cost, scost = 1.0 - sims, None
        if self.states is not None:
            assert query_state is not None, "this aligner reads robot states: pass the call's state"
            scost = state_cost(self.states, query_state.to(self.keys.device)[None, None], self.state_sigma)
            cost = (1.0 - self.state_weight) * cost + self.state_weight * scost
        return sims, cost.masked_fill(~self.valid, float("inf")), scost

    def step(self, query: torch.Tensor, query_state: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """query [D] unit key -> per memory: aligned position [P], similarity there [P], mean alignment cost [P]."""
        sims, cost, scost = self.local_cost(query, query_state)
        if self.J is None:
            J = cost
        else:
            best = torch.full_like(self.J, float("inf"))
            for d in range(self.vmax + 1):
                if d >= self.J.shape[1]:
                    break
                shifted = self.J if d == 0 else F.pad(self.J[:, :-d], (d, 0), value=float("inf"))
                best = torch.minimum(best, shifted + self.gamma * abs(d - 1))
            J = cost + best
        self.J = J
        self.calls += 1
        ends = J.argmin(dim=1)
        score = J.gather(1, ends[:, None])[:, 0] / self.calls
        self.state_cost = None if scost is None else scost.gather(1, ends[:, None])[:, 0]
        return ends, sims.gather(1, ends[:, None])[:, 0], score


class FrameNN(ProgressiveAlignment):
    """
    Frame nearest neighbour, without history (RTCF's Frame-NN ablation): per memory, the most similar step to the
    current call alone; its cost 1 - similarity ranks the memories (with robot states: the step of lowest local cost,
    as in ProgressiveAlignment). Same interface as ProgressiveAlignment.
    """

    def step(self, query: torch.Tensor, query_state: Optional[torch.Tensor] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls += 1
        if self.states is None:
            sims = torch.matmul(self.keys, query.float().to(self.keys.device)).clamp(-1.0, 1.0)
            best, ends = sims.masked_fill(~self.valid, -2.0).max(dim=1)
            return ends, best, 1.0 - best
        sims, cost, scost = self.local_cost(query, query_state)
        best, ends = cost.min(dim=1)
        self.state_cost = scost.gather(1, ends[:, None])[:, 0]
        return ends, sims.gather(1, ends[:, None])[:, 0], best


ALIGNERS = {"pma": ProgressiveAlignment, "nn": FrameNN}


def make_aligner(method: str, keys: torch.Tensor, lengths: torch.Tensor, vmax: int, gamma: float,
                 states: Optional[torch.Tensor] = None, state_weight: float = 0.0, state_sigma: float = 0.3):
    assert method in ALIGNERS, f"Unknown alignment {method}"
    return ALIGNERS[method](keys, lengths, vmax, gamma, states=states, state_weight=state_weight,
                            state_sigma=state_sigma)


def align_sequence(query_keys: torch.Tensor, keys: torch.Tensor, lengths: torch.Tensor, vmax: int, gamma: float,
                   method: str = "pma", query_states: Optional[torch.Tensor] = None,
                   states: Optional[torch.Tensor] = None, state_weight: float = 0.0, state_sigma: float = 0.3):
    """
    Offline alignment, call by call as at deployment: query_keys [T, D] -> ends, similarities, costs, each [T, P];
    with the memories' `states` [P, L, Ds] (and the query's [T, Ds]) also the state costs at the aligned positions.
    """
    aligner = make_aligner(method, keys, lengths, vmax, gamma, states, state_weight, state_sigma)
    if states is None:
        steps = [aligner.step(q) for q in query_keys]
        return tuple(torch.stack(x) for x in zip(*steps))
    assert query_states is not None and len(query_states) == len(query_keys), "one robot state per query call"
    steps = [(*aligner.step(q, s), aligner.state_cost) for q, s in zip(query_keys, query_states)]
    return tuple(torch.stack(x) for x in zip(*steps))


# === Retrieval-key transform ===
def fit_key_transform(X: torch.Tensor, kind: str = "pca", dim: int = 128) -> Optional[dict]:
    """
    Fit on raw key vectors X [n, d] (the library's calls): `none`; `center` (subtract the mean); `pca` (center, then
    project on the top `dim` principal directions). The host's raw keys are nearly constant (one large shared
    component), which centering removes; see pace/key_study.py.
    """
    if kind == "none":
        return None
    X = X.float()
    mean = X.mean(dim=0)
    if kind == "center":
        return {"kind": "center", "mean": mean.contiguous()}
    assert kind == "pca", f"Unknown key transform {kind}"
    q = min(dim + 10, min(X.shape))
    _, _, V = torch.pca_lowrank(X - mean, q=q, center=False, niter=4)
    return {"kind": "pca", "mean": mean.contiguous(), "proj": V[:, :dim].contiguous()}


def apply_key_transform(x: torch.Tensor, transform: Optional[dict]) -> torch.Tensor:
    """Raw key vectors [..., d] -> unit keys [..., d'] (the transform's tensors are moved to x's device)."""
    x = x.float()
    if transform is not None:
        x = x - transform["mean"].to(x.device)
        if "proj" in transform:
            x = x @ transform["proj"].to(x.device)
    return F.normalize(x, dim=-1)


def transform_id(transform: Optional[dict]) -> str:
    """Short content hash, recorded in store info so keys from different fits are never mixed."""
    if transform is None:
        return "none"
    import hashlib

    h = hashlib.sha1(transform["kind"].encode())
    for name in ("mean", "proj"):
        if name in transform:
            h.update(transform[name].float().cpu().numpy().tobytes())
    return f"{transform['kind']}-{h.hexdigest()[:12]}"


# === Adapter ===
class GatedCrossAttention(nn.Module):
    """
    Action tokens read the reference tokens; `residual` is what the host adds to its hidden states (pre-norm: the
    norm is on the branch input only, the host's residual stream is never normalized here). The output projection is
    zero-initialized, so the residual is exactly 0 at initialization while every weight of the projection learns at
    full speed from the first step (a scalar tanh gate would hold the others' gradients near 0).

    With `cond_dim` the branch is conditioned on the host's condition vector (pi0.5: the flow-time embedding it feeds
    adaRMS), adaLN style: shift and scale of the normalized input and a multiplicative gate (1 + g) on the output,
    all from one zero-initialized linear map, so at initialization it equals the unconditioned branch.
    """

    def __init__(self, dim: int, num_heads: int, cond_dim: Optional[int] = None, null_value: bool = False,
                 out_bias: bool = True) -> None:
        """
        `null_value`: the first token (the adapter's null token) has a zero value, so attention on it adds nothing;
        with `out_bias=False` as well, a sample whose attention sits on the null token gets exactly 0 (the relevance
        gate of a v6 adapter can then switch its references off).
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.null_value = bool(null_value)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.o_proj = nn.Linear(dim, dim, bias=out_bias)
        nn.init.zeros_(self.o_proj.weight)
        if out_bias:
            nn.init.zeros_(self.o_proj.bias)
        if cond_dim:
            self.cond_proj = nn.Linear(cond_dim, 3 * dim)  # -> shift, scale, gate
            nn.init.zeros_(self.cond_proj.weight)
            nn.init.zeros_(self.cond_proj.bias)
        else:
            self.cond_proj = None

    def residual(self, x, tokens, attend, active, cond=None, bias=None):
        """
        x [B, N, D]; tokens [B, M, D]; attend [B, M] bool (True = may be attended); active [B];
        cond [B, Dc] or [B, N, Dc] (required with cond_dim); bias [B, M] added to the attention logits (the
        relevance gate's log weights) -> [B, N, D].
        """
        x = x.to(self.q_proj.weight.dtype)
        B, N, D = x.shape
        M, H = tokens.shape[1], self.num_heads
        h = self.norm(x)
        gate = None
        if self.cond_proj is not None:
            assert cond is not None, "this adapter is time-conditioned but the host passed no condition"
            cond = cond.to(x.dtype)
            if cond.dim() == 2:
                cond = cond[:, None]
            shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
            h = h * (1 + scale) + shift
        q = self.q_proj(h).view(B, N, H, D // H).transpose(1, 2)
        k, v = self.kv_proj(tokens).view(B, M, 2, H, D // H).permute(2, 0, 3, 1, 4)
        if self.null_value:  # the null token (index 0) contributes nothing
            v = torch.cat([torch.zeros_like(v[:, :, :1]), v[:, :, 1:]], dim=2)
        mask = attend[:, None, None, :]
        if bias is not None:
            mask = torch.where(mask, bias[:, None, None, :].to(q.dtype), torch.finfo(q.dtype).min)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
        out = self.o_proj(out.transpose(1, 2).reshape(B, N, D))
        if gate is not None:
            out = out * (1 + gate)
        return active.to(out.dtype)[:, None, None] * out  # samples without references: exactly 0

    def gate_value(self) -> float:
        """Gain of the zero-initialized projection (RMS weight x sqrt(fan-in))."""
        w = self.o_proj.weight.detach().float()
        return float(w.pow(2).mean().sqrt() * math.sqrt(w.shape[1]))


class GenericResidual(nn.Module):
    """
    The generic branch of a v4 adapter (`generic="mlp"`): a pre-norm residual MLP of the action tokens alone, the same
    for every sample whether it has references or not, so what can be learned without reading references (a plain
    fine-tune of the host on the training data) is learned here and not through the reference branch. Zero-initialized
    output projection; with `cond_dim` it is conditioned on the host's flow-time embedding (adaLN shift / scale and a
    (1 + g) output gate, zero-initialized), since the correction of a velocity field depends on the flow time. The
    time embedding is a function of one scalar (the flow time), so its projection is low-rank (`cond_rank`, through a
    cond_rank-dimensional bottleneck; 0 = a full cond_dim x 3 dim map, 3.1M parameters per layer at 1024).
    """

    def __init__(self, dim: int, hidden: int, cond_dim: Optional[int] = None, cond_rank: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        if cond_dim:
            if cond_rank and cond_rank < cond_dim:  # -> shift, scale, gate through a bottleneck
                self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_rank), nn.Linear(cond_rank, 3 * dim))
            else:
                self.cond_proj = nn.Sequential(nn.Linear(cond_dim, 3 * dim))
            nn.init.zeros_(self.cond_proj[-1].weight)
            nn.init.zeros_(self.cond_proj[-1].bias)
        else:
            self.cond_proj = None

    def residual(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, N, D]; cond [B, Dc] or [B, N, Dc] (required with cond_dim) -> [B, N, D]."""
        x = x.to(self.fc1.weight.dtype)
        h, gate = self.norm(x), None
        if self.cond_proj is not None:
            assert cond is not None, "this generic branch is time-conditioned but the host passed no condition"
            cond = cond.to(x.dtype)
            if cond.dim() == 2:
                cond = cond[:, None]
            shift, scale, gate = self.cond_proj(cond).chunk(3, dim=-1)
            h = h * (1 + scale) + shift
        out = self.fc2(F.gelu(self.fc1(h)))
        return out * (1 + gate) if gate is not None else out

    def gate_value(self) -> float:
        w = self.fc2.weight.detach().float()
        return float(w.pow(2).mean().sqrt() * math.sqrt(w.shape[1]))


class ExperienceAdapter(nn.Module):
    """
    Encodes M references into tokens and injects them into the host's action network: cross-attention after the
    blocks listed in `inject_layers`, and (input_inject) the reference chunks, weighted by a softmax over a learned
    score, added to action token k's input through a zero-initialized projection of step k, so that copying a
    reference is a linear map from the first block on. The input dict (built by `stack_slots` / `ExperienceRuntime` /
    the training script) holds, for B samples:
        actions [B, M, F, A]  reference action chunk at the matched step (normalized as the host's training data)
        scalars [B, M, S]     `reference_scalars`
        success [B, M]        outcome of the reference episode (1 = success)
        valid   [B, M]        which slots hold a reference
    and, for the v2 encoder (`FEATURE_KEYS`):
        ref_vis   [B, M, V, Dv]  the reference's image tokens at the matched step (the host's pooled grid cells)
        ref_state [B, M, Ds]     the reference's robot state there (normalized as the host's input)
        cur_vis   [B, V, Dv]     the current call's image tokens, computed the same way
        cur_state [B, Ds]        the current robot state

    Encoders (`encoder`):
      v1  every reference contributes F tokens, one per step of its chunk: an MLP of the step's action plus an MLP of
          its scalars, a success embedding, a step and a slot embedding; references never see each other. The input
          injection weights references by a linear score of their scalars.
      v2  every reference contributes 1 + act_segments + V tokens: a summary token (MLP of its scalars, success, state
          and state minus the current state), its chunk cut into act_segments segments (MLP of each segment's
          actions) and its image tokens (shared projection, grid-position embedding). The current call adds 1 + V
          tokens (an MLP of its state, its image tokens). A small pre-norm transformer (enc_layers x enc_dim) attends
          over all of them, so references are compared with each other and with the current scene; the references'
          output tokens, projected to the host width, are what the cross-attention reads (the current call's own
          tokens are only context). The input injection weights references by a linear score of their summary
          outputs.
    In both a learned null token is always attendable, so a sample with few references never attends to an empty set,
    and every output projection is zero-initialized. `scalar_names` are the per-reference scalars the retrieval
    provides (SCALAR_NAMES; V3_SCALAR_NAMES for the v3 retrieval), so an adapter is trained for one retrieval.

    `generic` (v4): "none" (default: the adapter acts only on samples with references, as v1-v3) or "mlp": every
    injection layer also gets a reference-independent `GenericResidual` (width generic_dim, time-conditioned with
    generic_time) and the action tokens' input a learned per-step bias, applied to every sample with or without
    references (`layer_residual` / `input_residual` with generic=True). Then "no references" means the host plus the
    generic branch, and what references add over it is the reference branch's own contribution. The generic modules
    are created after all others, so the reference branch's initial weights are the same as without them.

    `relevance` (v6): "none" (default) or "gate": the adapter can decline its references. Each reference gets a weight
    g in (0, 1) from its summary (v2: the encoder's summary output; v1: its scalars, which hold the match quality) through
    a small MLP; log g is added to the cross-attention logits of all its tokens and to its input-injection score. The
    null token's value is zero and the output / injection projections have no bias, and the injection softmax has a
    learned null option, so when every g is small the reference branch adds (close to) nothing. The null token's
    attention logit gets log T (T tokens per reference), so it competes with a reference's T tokens as one option, as
    the injection's null does with one chunk: the same g closes both paths alike. Train it on top of a
    generic branch (two stages): there, declining a reference costs nothing, so the gate learns to open only for
    references that help.

    `position` (where in a host block the cross-attention sits) is interpreted by the host attachment and stored with
    the adapter, since an adapter is trained for one position. `time_cond` conditions every cross-attention on the
    host's condition vector of size `cond_dim` (default: hidden_size).
    """

    POSITIONS = ("post_layer", "pre_mlp")
    ENCODERS = ("v1", "v2")
    GENERICS = ("none", "mlp")
    RELEVANCES = ("none", "gate")
    FEATURE_KEYS = ("ref_vis", "ref_state", "cur_vis", "cur_state")

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 8,
        depth: int = 18,
        inject_every: int = 2,
        inject_layers: Optional[Sequence[int]] = None,
        action_dim: int = 7,
        chunk_len: int = 50,
        max_refs: int = 8,
        input_inject: bool = True,
        position: str = "post_layer",
        time_cond: bool = False,
        cond_dim: Optional[int] = None,
        encoder: str = "v1",
        vis_tokens: Optional[int] = None,
        vis_dim: Optional[int] = None,
        state_dim: Optional[int] = None,
        enc_dim: int = 512,
        enc_layers: int = 2,
        enc_heads: int = 8,
        act_segments: int = 10,
        scalar_names: Optional[Sequence[str]] = None,
        generic: str = "none",
        generic_dim: int = 256,
        generic_time: bool = True,
        generic_cond_rank: int = 64,
        relevance: str = "none",
        relevance_hidden: int = 64,
    ) -> None:
        super().__init__()
        assert position in self.POSITIONS, f"Unknown position {position}"
        assert encoder in self.ENCODERS, f"Unknown encoder {encoder}"
        assert generic in self.GENERICS, f"Unknown generic branch {generic}"
        assert relevance in self.RELEVANCES, f"Unknown relevance {relevance}"
        gated = relevance == "gate"
        self.scalar_names = tuple(scalar_names) if scalar_names else SCALAR_NAMES
        assert self.scalar_names in (SCALAR_NAMES, V3_SCALAR_NAMES), f"Unknown scalars {self.scalar_names}"
        n_scalars = len(self.scalar_names)
        if inject_layers is None:
            inject_layers = [i for i in range(depth) if (i + 1) % inject_every == 0]
        self.inject_layers = sorted({int(i) for i in inject_layers})
        assert self.inject_layers, "no injection layer"
        assert all(0 <= i < depth for i in self.inject_layers), f"inject_layers {self.inject_layers} vs depth {depth}"
        cond_dim = (cond_dim or hidden_size) if time_cond else None
        self.position, self.time_cond, self.encoder = position, bool(time_cond), encoder
        self.config = dict(
            hidden_size=hidden_size, num_heads=num_heads, depth=depth, inject_every=inject_every,
            inject_layers=self.inject_layers, action_dim=action_dim, chunk_len=chunk_len, max_refs=max_refs,
            input_inject=input_inject, position=position, time_cond=self.time_cond, cond_dim=cond_dim,
            encoder=encoder, scalar_names=list(self.scalar_names), generic=generic,
        )
        if generic != "none":
            self.config.update(generic_dim=int(generic_dim), generic_time=bool(generic_time),
                               generic_cond_rank=int(generic_cond_rank))
        if gated:
            self.config.update(relevance=relevance, relevance_hidden=int(relevance_hidden))
        if encoder == "v1":  # (creation order as before the v2 encoder existed: same initial weights per seed)
            self.step_mlp = nn.Sequential(nn.Linear(action_dim, hidden_size), nn.SiLU(),
                                          nn.Linear(hidden_size, hidden_size))
            self.step_emb = nn.Parameter(torch.randn(chunk_len, hidden_size) * 0.02)
            self.scalar_mlp = nn.Sequential(
                nn.Linear(n_scalars, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
            )
            self.success_emb = nn.Embedding(2, hidden_size)
            self.slot_emb = nn.Parameter(torch.randn(max_refs, hidden_size) * 0.02)
            self.null_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            self.token_norm = nn.LayerNorm(hidden_size)
            score_in = n_scalars
        else:
            assert vis_tokens and vis_dim and state_dim, "the v2 encoder needs vis_tokens, vis_dim and state_dim"
            assert enc_dim % enc_heads == 0, f"enc_dim {enc_dim} must be divisible by enc_heads {enc_heads}"
            self.config.update(vis_tokens=vis_tokens, vis_dim=vis_dim, state_dim=state_dim, enc_dim=enc_dim,
                               enc_layers=enc_layers, enc_heads=enc_heads, act_segments=act_segments)
            self.segment_len = math.ceil(chunk_len / act_segments)  # the chunk is padded with its last row to fit
            d = enc_dim
            mlp = lambda n_in: nn.Sequential(nn.Linear(n_in, d), nn.SiLU(), nn.Linear(d, d))  # noqa: E731
            self.vis_in = nn.Sequential(nn.LayerNorm(vis_dim), nn.Linear(vis_dim, d))
            self.vis_pos = nn.Parameter(torch.randn(vis_tokens, d) * 0.02)
            self.segment_mlp = mlp(self.segment_len * action_dim)
            self.segment_emb = nn.Parameter(torch.randn(act_segments, d) * 0.02)
            self.summary_mlp = mlp(n_scalars + 1 + 2 * state_dim)  # scalars, success, state, state - current
            self.current_mlp = mlp(state_dim)
            # token kinds: reference summary / action segment / image, current summary / image
            self.kind_emb = nn.Parameter(torch.randn(5, d) * 0.02)
            self.slot_emb = nn.Parameter(torch.randn(max_refs, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, enc_heads, dim_feedforward=4 * d, dropout=0.0, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False)
            self.out_norm = nn.LayerNorm(d)
            self.out_proj = nn.Linear(d, hidden_size)
            self.null_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
            score_in = d
        self.layers = nn.ModuleDict(
            {str(i): GatedCrossAttention(hidden_size, num_heads, cond_dim, null_value=gated, out_bias=not gated)
             for i in self.inject_layers}
        )
        if input_inject:
            self.inject_score = nn.Linear(score_in, 1)  # zero: references weighted uniformly at first
            nn.init.zeros_(self.inject_score.weight)
            nn.init.zeros_(self.inject_score.bias)
            # zero: no effect at initialization (v6: no bias, so a zero-weighted chunk adds exactly nothing)
            self.inject_proj = nn.Linear(action_dim, hidden_size, bias=not gated)
            nn.init.zeros_(self.inject_proj.weight)
            if not gated:
                nn.init.zeros_(self.inject_proj.bias)
        else:
            self.inject_score = self.inject_proj = None
        # v4 generic branch, created last (the modules above get the same initial weights as without it)
        self.generic = generic
        self.generic_time = bool(generic_time) and generic != "none"
        if generic == "mlp":
            self.generic_layers = nn.ModuleDict({
                str(i): GenericResidual(hidden_size, generic_dim, hidden_size if self.generic_time else None,
                                        generic_cond_rank)
                for i in self.inject_layers
            })
            self.generic_input = nn.Parameter(torch.zeros(chunk_len, hidden_size))  # per action step, zero at first
        else:
            self.generic_layers, self.generic_input = None, None
        # v6 relevance gate (created after everything else; part of the reference branch)
        self.relevance = relevance
        # of the last encode: mean / min / max g of its valid references, and the input injection's weight on them
        self.relevance_stats: Dict[str, float] = {}
        if gated:
            self.relevance_head = nn.Sequential(nn.Linear(score_in, relevance_hidden), nn.SiLU(),
                                                nn.Linear(relevance_hidden, 1))
            nn.init.zeros_(self.relevance_head[-1].weight)
            nn.init.zeros_(self.relevance_head[-1].bias)  # every reference starts at g = 0.5
            self.inject_null = nn.Parameter(torch.zeros(())) if input_inject else None  # injection's "none" score
        else:
            self.relevance_head, self.inject_null = None, None

    @property
    def has_generic(self) -> bool:
        return self.generic_layers is not None

    def generic_parameters(self) -> List[nn.Parameter]:
        if not self.has_generic:
            return []
        return [*self.generic_layers.parameters(), self.generic_input]

    def reference_parameters(self) -> List[nn.Parameter]:
        ids = {id(p) for p in self.generic_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    @property
    def uses_features(self) -> bool:
        """Whether the adapter reads image tokens and robot states (`FEATURE_KEYS`)."""
        return self.encoder == "v2"

    def _relevance(self, score_input, valid):
        """v6: log g [B, M] of every reference from its summary / scalars (None without a relevance gate)."""
        if self.relevance_head is None:
            return None
        logit = self.relevance_head(score_input)[..., 0]
        g = torch.sigmoid(logit.detach().float())[valid]
        self.relevance_stats = {}
        if g.numel():
            self.relevance_stats = {"relevance_mean": float(g.mean()), "relevance_min": float(g.min()),
                                    "relevance_max": float(g.max())}
        return F.logsigmoid(logit)

    def encode(self, exp: Dict[str, torch.Tensor]):
        """
        -> (tokens [B, 1 + M*T, H], attend [B, 1 + M*T], active [B], input residual [B, F, H] or None, attention bias
        [B, 1 + M*T] or None), with T tokens per reference (v1: F; v2: 1 + act_segments + V); the bias (v6) is each
        reference's log relevance on its tokens and log T on the null token.
        """
        assert exp["scalars"].shape[-1] == len(self.scalar_names), \
            f"{exp['scalars'].shape[-1]} scalars per reference vs the adapter's {self.scalar_names} (another retrieval?)"
        if self.encoder == "v2":
            return self._encode_v2(exp)
        dtype = self.null_token.dtype
        actions = exp["actions"].to(dtype)
        B, M, Fl = actions.shape[:3]
        assert M <= self.slot_emb.shape[0], f"{M} references > max_refs {self.slot_emb.shape[0]}"
        assert Fl <= self.step_emb.shape[0], f"chunk of {Fl} steps > chunk_len {self.step_emb.shape[0]}"
        valid = exp["valid"].bool()
        per_ref = self.scalar_mlp(exp["scalars"].to(dtype)) + self.success_emb(exp["success"].long())  # [B, M, H]
        h = self.step_mlp(actions) + per_ref[:, :, None] + self.step_emb[:Fl]  # [B, M, F, H]
        h = (self.token_norm(h) + self.slot_emb[:M, None]).flatten(1, 2)  # [B, M * F, H]
        token_valid = valid[:, :, None].expand(B, M, Fl).flatten(1)
        tokens = torch.cat([self.null_token.expand(B, 1, -1), h], dim=1)
        attend = torch.cat([torch.ones_like(token_valid[:, :1]), token_valid], dim=1)  # the null token: always
        active = valid.any(dim=1)
        scalars = exp["scalars"].to(dtype)
        log_g = self._relevance(scalars, valid)
        bias = self._attention_bias(log_g, Fl)
        return tokens, attend, active, self._input_residual(scalars, actions, valid, active, log_g), bias

    @staticmethod
    def _attention_bias(log_g, tokens_per_ref: int):
        """
        [B, M] log relevance -> [B, 1 + M * T] attention bias, or None. The null token gets log T: with equal logits a
        reference's T tokens together then weigh g against the null's 1, as in the input injection's softmax (with 0
        and 4 references of 43 tokens, the null took ~1% of the attention at g = 0.5, and cutting the references to
        10% took g ~ 7e-4 there against ~0.03 in the injection).
        """
        if log_g is None:
            return None
        per_token = log_g[:, :, None].expand(-1, -1, tokens_per_ref).flatten(1)
        null = torch.full_like(per_token[:, :1], math.log(tokens_per_ref))
        return torch.cat([null, per_token], dim=1)

    def _input_residual(self, score_input, actions, valid, active, log_g=None) -> Optional[torch.Tensor]:
        """
        Score-weighted reference chunk through the zero-initialized input projection: [B, F, H], or None. v6: the
        score adds log g, and a learned null option takes part in the softmax (its chunk is zero).
        """
        if self.inject_proj is None:
            return None
        score = self.inject_score(score_input)[..., 0]
        if log_g is not None:
            score = score + log_g
        score = score.masked_fill(~valid, float("-inf"))  # [B, M]
        if self.inject_null is not None:
            null = self.inject_null.to(score.dtype).expand(score.shape[0], 1)
            weights = torch.softmax(torch.cat([score, null], dim=1), dim=1)[:, :-1]
            mass = weights.detach().float().sum(dim=1)[active]  # how much of the injection the references take
            if mass.numel():
                self.relevance_stats["inject_ref_mass"] = float(mass.mean())
        else:
            weights = torch.nan_to_num(torch.softmax(score, dim=1))
        chunk = (weights[..., None, None] * actions).sum(dim=1)  # [B, F, A]
        # rows without references: exactly 0 (a zero chunk alone would still add the projection's trained bias)
        return self.inject_proj(chunk) * active.to(chunk.dtype)[:, None, None]

    def _encode_v2(self, exp: Dict[str, torch.Tensor]):
        dtype = self.null_token.dtype
        missing = [k for k in self.FEATURE_KEYS if k not in exp]
        assert not missing, f"the v2 encoder needs {missing} (a store / library built with features)"
        actions = exp["actions"].to(dtype)
        B, M, Fl, A = actions.shape
        assert M <= self.slot_emb.shape[0], f"{M} references > max_refs {self.slot_emb.shape[0]}"
        S, L = self.segment_emb.shape[0], self.segment_len
        assert Fl <= S * L, f"chunk of {Fl} steps > {S} segments x {L}"
        valid = exp["valid"].bool()
        active = valid.any(dim=1)
        kind = self.kind_emb
        # action segments: [B, M, S, d]
        padded = torch.cat([actions, actions[:, :, -1:].expand(B, M, S * L - Fl, A)], dim=2) if S * L > Fl else actions
        segments = self.segment_mlp(padded.reshape(B, M, S, L * A)) + self.segment_emb + kind[1]
        # summary: [B, M, 1, d]
        ref_state, cur_state = exp["ref_state"].to(dtype), exp["cur_state"].to(dtype)
        summary_in = torch.cat([exp["scalars"].to(dtype), exp["success"].to(dtype)[..., None], ref_state,
                                ref_state - cur_state[:, None]], dim=-1)
        summary = (self.summary_mlp(summary_in) + kind[0])[:, :, None]
        # image tokens: [B, M, V, d] and [B, V, d]
        ref_vis = self.vis_in(exp["ref_vis"].to(dtype)) + self.vis_pos + kind[2]
        cur_vis = self.vis_in(exp["cur_vis"].to(dtype)) + self.vis_pos + kind[4]
        per_ref = torch.cat([summary, segments, ref_vis], dim=2) + self.slot_emb[:M, None]  # [B, M, T, d]
        T = per_ref.shape[2]
        current = torch.cat([(self.current_mlp(cur_state) + kind[3])[:, None], cur_vis], dim=1)  # [B, 1 + V, d]
        sequence = torch.cat([per_ref.flatten(1, 2), current], dim=1)
        ref_valid = valid[:, :, None].expand(B, M, T).flatten(1)  # [B, M*T]
        keep = torch.cat([ref_valid, ref_valid.new_ones(B, current.shape[1])], dim=1)  # current tokens: always
        out = self.context(sequence, src_key_padding_mask=~keep)[:, : M * T]  # [B, M*T, d]
        tokens = torch.cat([self.null_token.expand(B, 1, -1), self.out_proj(self.out_norm(out))], dim=1)
        attend = torch.cat([torch.ones_like(ref_valid[:, :1]), ref_valid], dim=1)
        summary_out = out.view(B, M, T, -1)[:, :, 0]  # [B, M, d]
        log_g = self._relevance(summary_out, valid)
        return (tokens, attend, active, self._input_residual(summary_out, actions, valid, active, log_g),
                self._attention_bias(log_g, T))

    def layer_residual(self, layer_index: int, x: torch.Tensor, encoded, cond=None,
                       generic: bool = True) -> Optional[torch.Tensor]:
        """
        Residual to add at host block `layer_index`: the reference branch's (`encoded` from `encode`, None = no
        references) plus, with `generic` and a generic branch, the generic branch's; None if nothing acts there.
        """
        key, out = str(layer_index), None
        if encoded is not None and key in self.layers:
            tokens, attend, active = encoded[:3]
            bias = encoded[4] if len(encoded) > 4 else None
            out = self.layers[key].residual(x, tokens, attend, active, cond=cond if self.time_cond else None,
                                            bias=bias)
        if generic and self.has_generic and key in self.generic_layers:
            g = self.generic_layers[key].residual(x, cond if self.generic_time else None)
            out = g if out is None else out + g
        return out

    def input_residual(self, encoded, num_tokens: int, generic: bool = True) -> Optional[torch.Tensor]:
        """
        Residual to add to the action tokens' input [B or 1, F, H]: the reference branch's input injection (None
        without references or input_inject) plus, with `generic` and a generic branch, its per-step bias.
        """
        out = encoded[3] if encoded is not None else None
        if generic and self.has_generic:
            assert num_tokens <= self.generic_input.shape[0], f"{num_tokens} action tokens > chunk_len"
            bias = self.generic_input[:num_tokens][None]
            out = bias if out is None else out + bias.to(out.dtype)
        return out

    def generic_gates(self) -> Dict[str, float]:
        """Gains of the generic branch (empty without one)."""
        if not self.has_generic:
            return {}
        out = {f"generic_{k}": layer.gate_value() for k, layer in self.generic_layers.items()}
        out["generic_input"] = float(self.generic_input.detach().float().pow(2).mean().sqrt())
        return out

    def gates(self) -> Dict[str, float]:
        out = {f"gate_{k}": layer.gate_value() for k, layer in self.layers.items()}
        if self.inject_proj is not None:
            w = self.inject_proj.weight.detach().float()
            out["gate_input"] = float(w.pow(2).mean().sqrt() * math.sqrt(w.shape[1]))
        return out


def save_adapter(adapter: ExperienceAdapter, path, **extra) -> None:
    torch.save({"config": adapter.config, "state_dict": adapter.state_dict(), **extra}, path)


def load_adapter(path) -> Tuple[ExperienceAdapter, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(ckpt["config"])
    adapter = ExperienceAdapter(**config)
    adapter.load_state_dict(ckpt["state_dict"])
    return adapter, ckpt


# === Store ===
def make_episode(
    task_key: str,
    keys: torch.Tensor,  # [T, D] unit vectors
    actions: torch.Tensor,  # [T, F, A] normalized action chunks
    sig: Optional[torch.Tensor],  # [Ds] unit first-frame signature
    success: bool,
    source: str,
    name: str,
    meta: Optional[dict] = None,
    vis: Optional[torch.Tensor] = None,  # [T, V, Dv] image tokens per call (v2 encoder)
    state: Optional[torch.Tensor] = None,  # [T, Ds] normalized robot state per call (v2 encoder, v3 retrieval)
) -> dict:
    assert keys.shape[0] == actions.shape[0], f"{keys.shape[0]} keys vs {actions.shape[0]} action chunks"
    assert vis is None or state is not None, "image tokens are stored with the robot states"
    episode = {
        "name": name, "task_key": task_key, "keys": keys.half().cpu(), "actions": actions.float().cpu(),
        "sig": None if sig is None else sig.half().cpu(), "success": bool(success), "source": source,
        "meta": dict(meta or {}),
    }
    if state is not None:
        assert state.shape[0] == keys.shape[0], f"{name}: {state.shape[0]} states vs {keys.shape[0]} keys"
        episode["state"] = state.float().cpu()
    if vis is not None:
        assert vis.shape[0] == keys.shape[0], f"{name}: {vis.shape[0]} image token sets vs {keys.shape[0]} keys"
        episode["vis"] = vis.half().cpu()
    return episode


def has_features(episode: dict) -> bool:
    """Per-call image tokens and robot states (what the v2 encoder reads)."""
    return "vis" in episode and "state" in episode


def has_states(episode: dict) -> bool:
    """Per-call robot states (what the v3 retrieval aligns on)."""
    return "state" in episode


class ExperienceStore:
    """Episodes grouped by task key. Episodes are never modified once added, so stores can share them."""

    def __init__(self, episodes: Optional[Sequence[dict]] = None, info: Optional[dict] = None,
                 key_transform: Optional[dict] = None) -> None:
        self.episodes: List[dict] = []
        self.by_task: Dict[str, List[int]] = {}
        self.info = dict(info or {})  # how the keys were computed (checked against the host at attach time)
        self.key_transform = key_transform  # fitted on the library (fit_key_transform); info["transform"] is its id
        self._sigs: Optional[tuple] = None  # (episode count, stacked signatures, has-signature mask) cache
        for episode in episodes or []:
            self.add(episode)

    def add(self, episode: dict) -> int:
        index = len(self.episodes)
        self.episodes.append(episode)
        self.by_task.setdefault(episode["task_key"], []).append(index)
        return index

    def copy(self) -> "ExperienceStore":
        return ExperienceStore(self.episodes, self.info, self.key_transform)

    def __len__(self) -> int:
        return len(self.episodes)

    def candidates(self, task_key: str, sources=("demo", "rollout"), success_only: bool = True,
                   exclude: Sequence[str] = ()) -> List[int]:
        exclude = set(exclude)
        return [
            i for i in self.by_task.get(task_key, [])
            if self.episodes[i]["source"] in sources and (self.episodes[i]["success"] or not success_only)
            and self.episodes[i]["name"] not in exclude
        ]

    def select(self, task_key: str, k: int, strategy: str = "similar", sig: Optional[torch.Tensor] = None,
               rng: Optional[np.random.Generator] = None, **filters) -> List[int]:
        """
        Up to k episodes of the task. `similar`: highest first-frame signature similarity (falls back to `recent`
        without signatures); `recent`: the last added; `random`: uniform without replacement.
        """
        pool = self.candidates(task_key, **filters)
        if not pool or k <= 0:
            return []
        if strategy == "similar" and sig is not None and all(self.episodes[i]["sig"] is not None for i in pool):
            sigs = torch.stack([self.episodes[i]["sig"] for i in pool]).float()
            order = torch.argsort(sigs @ sig.float().cpu(), descending=True).tolist()
            return [pool[o] for o in order[:k]]
        if strategy == "random":
            rng = rng or np.random.default_rng()
            return [pool[i] for i in rng.permutation(len(pool))[:k]]
        return pool[-k:][::-1]

    def scene_similarity(self, sig: torch.Tensor, tasks: Sequence[str], **filters) -> torch.Tensor:
        """
        Per task, the highest first-frame signature similarity of its candidate episodes to `sig` [K] (-inf for a task
        without candidates or signatures). The stacked signatures are cached until episodes are added.
        """
        n = len(self.episodes)
        if self._sigs is None or self._sigs[0] != n:
            have = [ep["sig"] is not None for ep in self.episodes]
            dim = next((ep["sig"].shape[0] for ep in self.episodes if ep["sig"] is not None), 0)
            stacked = torch.zeros(n, dim)
            for i, ep in enumerate(self.episodes):
                if have[i]:
                    stacked[i] = ep["sig"].float()
            self._sigs = (n, stacked, torch.tensor(have, dtype=torch.bool))
        _, stacked, have = self._sigs
        sims = stacked @ sig.float().cpu() if stacked.shape[1] else torch.zeros(n)
        sims = sims.masked_fill(~have, float("-inf"))
        out = torch.full((len(tasks),), float("-inf"))
        for k, task in enumerate(tasks):
            pool = self.candidates(task, **filters)
            if pool:
                out[k] = sims[torch.tensor(pool)].max()
        return out

    def save(self, path) -> None:
        # version 4: episodes may hold per-call image tokens and robot states ("vis", "state"; v2 encoder)
        torch.save({"version": 4, "info": self.info, "key_transform": self.key_transform, "episodes": self.episodes},
                   path)

    @classmethod
    def load(cls, paths: Sequence, mmap: bool = True) -> "ExperienceStore":
        """
        Stores saved by `save`. With `mmap` the tensors stay memory-mapped from the file (read on access, the page
        cache shared by every process that loads it), so a library larger than RAM (image tokens of all
        lerobot/libero_plus demos: ~30 GB) costs no resident memory per training process.
        """
        if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
            paths = [paths]
        store = cls()
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False, mmap=mmap)
            info = data.get("info", {})
            if store.info and info and info != store.info:
                raise ValueError(f"{path}: key info {info} differs from {store.info}")
            if not store.info:
                store.info, store.key_transform = dict(info), data.get("key_transform")
            for episode in data["episodes"]:
                store.add(episode)
        return store

    def summary(self) -> Dict[str, Dict[str, int]]:
        out = {}
        for task, indices in self.by_task.items():
            counts = collections.Counter(
                f"{self.episodes[i]['source']}_{'success' if self.episodes[i]['success'] else 'failure'}"
                for i in indices
            )
            out[task] = dict(counts)
        return out


# === Reference features ===
def reference_slot(ref: dict, j: int, similarity: float, align_cost: float, call: int, features: bool = False,
                   extra: Sequence[float] = ()) -> dict:
    """One reference read at its step j while the current episode is at call `call` (`extra`: v3 scalars)."""
    T = ref["actions"].shape[0]
    slot = {"actions": ref["actions"][j],
            "scalars": reference_scalars(j, similarity, align_cost, call, T) + [float(x) for x in extra],
            "success": int(ref["success"])}
    if features:
        slot["ref_vis"], slot["ref_state"] = ref["vis"][j], ref["state"][j]
    return slot


def stack_slots(slots: List[dict], max_refs: int, chunk_shape,
                feature_shapes: Optional[Tuple[Tuple[int, int], int]] = None,
                current: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                n_scalars: int = len(SCALAR_NAMES)) -> Dict[str, torch.Tensor]:
    """
    One sample's references (`reference_slot`) -> padded [1, max_refs, ...] tensors (CPU). With `feature_shapes`
    ((V, Dv), Ds) also the v2 encoder's inputs: the references' image tokens and states, and `current` = (image tokens
    [V, Dv], state [Ds]) of the current call (zeros without it).
    """
    actions = torch.zeros(1, max_refs, *chunk_shape)
    scalars = torch.zeros(1, max_refs, n_scalars)
    success = torch.zeros(1, max_refs, dtype=torch.long)
    valid = torch.zeros(1, max_refs, dtype=torch.bool)
    out = {"actions": actions, "scalars": scalars, "success": success, "valid": valid}
    if feature_shapes is not None:
        (V, Dv), Ds = feature_shapes
        out.update(ref_vis=torch.zeros(1, max_refs, V, Dv, dtype=torch.half), ref_state=torch.zeros(1, max_refs, Ds),
                   cur_vis=torch.zeros(1, V, Dv, dtype=torch.half), cur_state=torch.zeros(1, Ds))
        if current is not None:
            out["cur_vis"][0], out["cur_state"][0] = current[0], current[1]
    for m, slot in enumerate(slots[:max_refs]):
        assert len(slot["scalars"]) == n_scalars, f"{len(slot['scalars'])} scalars vs {n_scalars}"
        actions[0, m], scalars[0, m], success[0, m], valid[0, m] = (slot["actions"], torch.tensor(slot["scalars"]),
                                                                    slot["success"], True)
        if feature_shapes is not None:
            out["ref_vis"][0, m], out["ref_state"][0, m] = slot["ref_vis"], slot["ref_state"]
    return out


def empty_slots(batch_size: int, max_refs: int, chunk_shape, feature_shapes=None,
                n_scalars: int = len(SCALAR_NAMES)) -> Dict[str, torch.Tensor]:
    out = stack_slots([], max_refs, chunk_shape, feature_shapes, n_scalars=n_scalars)
    return {k: v.expand(batch_size, *v.shape[1:]).clone() for k, v in out.items()}


def executed_chunks(predicted: List[np.ndarray], executed_per_call: int, chunk_len: int) -> np.ndarray:
    """
    Per-call chunks of what a rollout executed: call k's chunk is the first `executed_per_call` predicted actions of
    calls k, k+1, ... concatenated; past the last executed action it continues with the last call's unexecuted
    prediction, then repeats the final row. predicted: list of [F, A] normalized chunks -> [T, chunk_len, A].
    """
    executed = np.concatenate([p[:executed_per_call] for p in predicted], axis=0)
    tail = np.concatenate([executed, predicted[-1][executed_per_call:]], axis=0)
    out = []
    for k in range(len(predicted)):
        chunk = tail[k * executed_per_call : k * executed_per_call + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], chunk_len - len(chunk), axis=0)], axis=0)
        out.append(chunk)
    return np.stack(out).astype(np.float32)


class ExperienceRuntime:
    """
    State of one running episode. On the first call it gathers the candidate pool (the task's successful episodes,
    narrowed to the `pool_size` most similar first frames when there are more); at every call it advances the
    alignment (`align`: PMA over the whole history, or frame nearest neighbour) over the whole pool and returns the
    `num_refs` best-aligned candidates as adapter inputs. It also records the episode
    (keys, predicted chunks, signature) so it can be added to a store at the end.
    Conditions: `real`; `unaligned` (control: a random step of each chosen reference); `other_task` (control:
    `num_refs` random successful episodes of other tasks, aligned the same way); `none`.
    With `features` (an adapter with the v2 encoder) every call also passes its image tokens and robot state: the
    adapter inputs then carry the references' and the current call's, every pool episode must have them (a store
    built with features), and the recorded episode keeps them.

    `retrieval` v3 (`task_key` is then the instruction the policy is told, `true_task` the env's task):
      - candidates: not only the episodes stored under the exact instruction but those of the `text_tasks` tasks
        whose instruction is closest to `query_text` (`task_text(query, tasks)` -> similarities) and of the
        `scene_tasks` tasks whose first frames are closest to this one (`choose_tasks`; without an instruction, e.g.
        query_text None, the text_tasks + scene_tasks closest scenes);
      - alignment: every call also passes its robot state; the local cost is (1 - state_weight) * (1 - cos) +
        state_weight * state cost (`state_cost`, scale state_sigma), so a changed camera or lighting moves only part
        of it;
      - ranking: mean alignment cost + text_weight * (1 - instruction similarity of the candidate's task);
      - the adapter also reads V3_SCALAR_NAMES' extra scalars (task similarity, whether an instruction was used, the
        state cost at the matched step), so it can discount a reference the retrieval is unsure about.
    `other_task` never draws from `true_task`.
    """

    def __init__(
        self,
        store: ExperienceStore,
        task_key: str,
        condition: str = "real",
        num_refs: int = 4,
        vmax: int = 3,
        gamma: float = 0.01,
        align: str = "pma",
        pool_size: int = 64,
        select: str = "similar",
        sources=("demo", "rollout"),
        success_only: bool = True,
        max_refs: int = 8,
        chunk_shape=(50, 7),
        seed: int = 0,
        device="cuda",
        features: bool = False,
        retrieval: str = "classic",
        state_weight: float = 0.0,
        state_sigma: float = 0.3,
        text_weight: float = 0.0,
        text_tasks: int = 3,
        scene_tasks: int = 2,
        task_text=None,
        query_text: Optional[str] = None,
        true_task: Optional[str] = None,
        n_scalars: int = len(SCALAR_NAMES),
        max_image_cost: Optional[float] = None,
    ) -> None:
        """
        `max_image_cost` (v6's hard gate): a reference whose image cost (1 - key similarity) at its matched step
        exceeds it is not read at that call; a call left without references runs the host (plus a generic branch).
        """
        assert condition in CONDITIONS, f"Unknown condition {condition}"
        assert retrieval in RETRIEVALS, f"Unknown retrieval {retrieval}"
        assert n_scalars == len(V3_SCALAR_NAMES if retrieval == "v3" else SCALAR_NAMES), \
            f"the {retrieval} retrieval gives {len(V3_SCALAR_NAMES if retrieval == 'v3' else SCALAR_NAMES)} scalars " \
            f"per reference, the adapter reads {n_scalars}: it was trained for another retrieval"
        self.store, self.task_key, self.condition = store, task_key, condition
        self.true_task = true_task if true_task is not None else task_key  # other_task control: never this task
        self.features = bool(features)
        self.retrieval = retrieval
        self.keep_state = self.features or retrieval == "v3"  # the v3 retrieval aligns on robot states too
        self.state_weight, self.state_sigma, self.text_weight = float(state_weight), float(state_sigma), text_weight
        self.text_tasks, self.scene_tasks, self.n_scalars = int(text_tasks), int(scene_tasks), int(n_scalars)
        self.task_text, self.query_text = task_text, query_text  # v3: instruction similarity (query, tasks) -> [K]
        self.text_known = False
        self.prior: Optional[torch.Tensor] = None  # v3: ranking penalty of each pool episode [P]
        self.member_text: Optional[torch.Tensor] = None  # v3: instruction similarity of each pool episode's task [P]
        self.vis: List[torch.Tensor] = []  # per call [V, Dv] (features only)
        self.states: List[torch.Tensor] = []  # per call [Ds] (features or the v3 retrieval)
        assert align in ALIGNERS, f"Unknown alignment {align}"
        self.num_refs, self.vmax, self.gamma, self.align = min(num_refs, max_refs), vmax, gamma, align
        self.pool_size, self.select, self.sources = pool_size, select, tuple(sources)
        self.success_only = success_only
        self.max_refs, self.chunk_shape, self.device = max_refs, tuple(chunk_shape), device
        self.rng = np.random.default_rng(seed)
        self.pool: List[dict] = []
        self.pma = None  # the aligner (ProgressiveAlignment or FrameNN)
        self.call = 0
        self.max_image_cost = None if max_image_cost is None else float(max_image_cost)
        self.declined = 0  # calls whose references the hard gate turned away
        self.keys: List[torch.Tensor] = []
        self.predicted: List[np.ndarray] = []
        self.sig: Optional[torch.Tensor] = None
        self.trace: List[List[Tuple[int, int, float, float]]] = []  # per call: (pool index, j, similarity, cost)

    def _build_pool(self) -> None:
        filters = dict(sources=self.sources, success_only=self.success_only)
        prior = None
        if self.condition == "other_task":
            others = [i for task in sorted(self.store.by_task) if task != self.true_task
                      for i in self.store.candidates(task, **filters)]
            picked = [others[i] for i in self.rng.permutation(len(others))[: self.num_refs]]
        elif self.retrieval == "v3":
            picked, prior = self._v3_pool(filters)
        else:
            k = self.pool_size if self.pool_size > 0 else len(self.store)
            picked = self.store.select(self.task_key, k, self.select, sig=self.sig, rng=self.rng, **filters)
        self.pool = [self.store.episodes[i] for i in picked]
        if self.features:
            lacking = [ep["name"] for ep in self.pool if not has_features(ep)]
            if lacking:
                raise ValueError(f"{len(lacking)} store episodes (e.g. {lacking[0]}) have no image tokens / states, "
                                 "which this adapter's v2 encoder reads; build the library with --features 1")
        states = None
        if self.retrieval == "v3":
            lacking = [ep["name"] for ep in self.pool if not has_states(ep)]
            if lacking:
                raise ValueError(f"{len(lacking)} store episodes (e.g. {lacking[0]}) have no robot states, which the "
                                 "v3 retrieval aligns on; rebuild the library with this build_library")
            if self.member_text is None:  # other_task: the instruction similarity of the episodes' own tasks
                sims = self._text_sims([ep["task_key"] for ep in self.pool])
                self.text_known = sims is not None
                self.member_text = sims if sims is not None else torch.zeros(len(self.pool))
            if self.pool:
                states = pad_keys([ep["state"] for ep in self.pool])[0].to(self.device)
        self.prior = prior
        if self.pool:
            keys, lengths = pad_keys([ep["keys"] for ep in self.pool])
            self.pma = make_aligner(self.align, keys.to(self.device), lengths, self.vmax, self.gamma, states,
                                    self.state_weight, self.state_sigma)

    def _text_sims(self, tasks: Sequence[str]) -> Optional[torch.Tensor]:
        """Instruction similarity of the current instruction to each task [K], or None without an instruction."""
        if self.task_text is None or not self.query_text:
            return None
        return torch.as_tensor(self.task_text(self.query_text, list(tasks)), dtype=torch.float32)

    def _v3_pool(self, filters):
        """
        v3 candidates: the episodes of the tasks closest in instruction and in first-frame scene (`choose_tasks`),
        up to pool_size per task; their ranking prior is text_weight * (1 - instruction similarity of their task).
        """
        tasks = [t for t in sorted(self.store.by_task) if self.store.candidates(t, **filters)]
        if not tasks:
            return [], None
        text = self._text_sims(tasks)
        self.text_known = text is not None
        scene = self.store.scene_similarity(self.sig, tasks, **filters) if self.sig is not None else None
        k = self.pool_size if self.pool_size > 0 else len(self.store)
        picked, member_text = [], []
        for t in choose_tasks(text, scene, self.text_tasks, self.scene_tasks):
            chosen = self.store.select(tasks[t], k, self.select, sig=self.sig, rng=self.rng, **filters)
            picked += chosen
            member_text += [float(text[t]) if text is not None else 0.0] * len(chosen)
        self.member_text = torch.tensor(member_text, dtype=torch.float32)
        prior = self.text_weight * (1.0 - self.member_text) if self.text_known else torch.zeros(len(picked))
        return picked, prior

    def __call__(self, key: torch.Tensor, sig: Optional[torch.Tensor] = None, vis: Optional[torch.Tensor] = None,
                 state: Optional[torch.Tensor] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        key [D] unit vector of this call; sig [Ds] unit signature (used on the first call); with `features` also vis
        [V, Dv] image tokens, and with `features` or the v3 retrieval state [Ds] normalized robot state of this call
        -> adapter inputs.
        """
        key = key.float().flatten()
        if self.keep_state:
            assert state is not None, "this runtime needs the call's robot state"
            self.states.append(state.detach().float().flatten().cpu())
        if self.features:
            assert vis is not None, "this runtime needs the call's image tokens"
            self.vis.append(vis.detach().half().cpu())
        if self.call == 0:
            self.sig = None if sig is None else sig.float().flatten().half().cpu()
            if self.condition != "none":
                self._build_pool()
        self.keys.append(key.half().cpu())
        exp = None
        if self.pma is not None:
            v3 = self.retrieval == "v3"
            ends, sims, costs = (x.cpu() for x in self.pma.step(key, self.states[-1] if v3 else None))
            rank = costs + self.prior if self.prior is not None else costs
            slots, step_trace = [], []
            for p in torch.argsort(rank)[: self.num_refs].tolist():
                ref, j, sim, cost = self.pool[p], int(ends[p]), float(sims[p]), float(costs[p])
                if self.condition == "unaligned":  # control: a random step of the same reference
                    j = int(self.rng.integers(ref["keys"].shape[0]))
                    sim = float(ref["keys"][j].float() @ key.cpu())
                if self.max_image_cost is not None and 1.0 - sim > self.max_image_cost:
                    continue  # v6 hard gate: too unlike the current call
                extra = ()
                if v3:
                    scost = float(state_cost(ref["state"][j], self.states[-1], self.state_sigma))
                    extra = (float(self.member_text[p]) if self.text_known else 0.0, float(self.text_known), scost)
                slots.append(reference_slot(ref, j, sim, cost, self.call, self.features, extra))
                step_trace.append((p, j, sim, cost))
            if slots:
                self.trace.append(step_trace)
                exp = stack_slots(slots, self.max_refs, self.chunk_shape, self.feature_shapes(),
                                  (self.vis[-1], self.states[-1]) if self.features else None, n_scalars=self.n_scalars)
            elif self.pool:
                self.declined += 1
        self.call += 1
        return exp

    def feature_shapes(self):
        """((V, Dv), Ds) of the recorded features, or None without features."""
        if not self.features or not self.vis:
            return None
        return tuple(self.vis[-1].shape), int(self.states[-1].shape[0])

    def record_prediction(self, normalized_chunk: np.ndarray) -> None:
        self.predicted.append(np.asarray(normalized_chunk, dtype=np.float32).copy())

    def info(self) -> dict:
        out = {"pool_size": len(self.pool), "num_calls": self.call, "condition": self.condition, "align": self.align}
        if self.max_image_cost is not None:
            out["max_image_cost"] = self.max_image_cost
            out["declined_frac"] = self.declined / max(self.call, 1)  # calls the hard gate left without references
            out["ref_calls"] = len(self.trace)
        if self.trace:
            used = collections.Counter(p for step in self.trace for p, *_ in step)
            out["refs"] = [self.pool[p]["name"] for p, _ in used.most_common()]
            out["mean_similarity"] = float(np.mean([s for step in self.trace for _, _, s, _ in step]))
            out["mean_align_cost"] = float(np.mean([c for step in self.trace for *_, c in step]))
            top = [step[0][0] for step in self.trace]
            out["top_ref_switches"] = int(sum(a != b for a, b in zip(top, top[1:])))
            out["top_ref_rollout_frac"] = float(np.mean([self.pool[p]["source"] == "rollout" for p in top]))
            p, j = self.trace[-1][0][:2]
            out["final_top_progress"] = float(j / max(self.pool[p]["keys"].shape[0] - 1, 1))
            if self.retrieval == "v3":  # how often the references come from the env's own task
                def true(ep):
                    return normalize_instruction(ep["meta"].get("task", ep["task_key"]))

                out["retrieval"] = "v3"
                out["pool_tasks"] = len({ep["task_key"] for ep in self.pool})
                out["top_ref_same_task"] = float(np.mean([true(self.pool[p]) == self.true_task for p in top]))
                out["refs_same_task"] = float(np.mean([true(self.pool[p]) == self.true_task
                                                       for step in self.trace for p, *_ in step]))
                out["text_known"] = bool(self.text_known)
        return out

    def to_episode(self, success: bool, executed_per_call: int, num_calls: Optional[int] = None,
                   name: Optional[str] = None, meta: Optional[dict] = None) -> Optional[dict]:
        """The episode as a store entry, cut to its first `num_calls` calls (the calls made before it ended)."""
        n = min(len(self.keys), len(self.predicted))
        if num_calls is not None:
            n = min(n, num_calls)
        if n == 0:
            return None
        chunks = executed_chunks(self.predicted[:n], executed_per_call, self.chunk_shape[0])
        features = {}
        if self.keep_state:
            features["state"] = torch.stack(self.states[:n])
        if self.features:
            features["vis"] = torch.stack(self.vis[:n])
        return make_episode(
            self.task_key, torch.stack(self.keys[:n]), torch.from_numpy(chunks), self.sig, success, "rollout",
            name or f"rollout_{time.time_ns()}", meta, **features,
        )
