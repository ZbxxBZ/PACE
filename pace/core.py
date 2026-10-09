"""
PACE core: Progress-Aligned Cross-episode Experience, independent of the host policy.

Every stored episode keeps, per policy call, a retrieval key (a unit vector computed by the host policy from the
call's observation) and the normalized action chunk executed from that call on, plus its outcome and a first-frame
scene signature. The current episode is aligned to every candidate stored episode of its task with Progressive Memory
Alignment (PMA, from RTCF, arXiv 2608.04527): a causal subsequence-DTW frontier over the whole execution history, so
frames of different stages that look alike are told apart by what came before. At every call the best-aligned
episodes are the references. What a reference did from its aligned position (its action chunk, one token per step),
where it is (progress, similarity, mean alignment cost, call indices) and how it ended (success) become reference
tokens. The host's action network reads them through cross-attention layers inserted after its blocks, whose output
projection is zero-initialized, and the score-weighted reference chunk is added, step by step, to the action tokens'
input through a zero-initialized projection. A sample without references gets an exactly zero residual, so the host
policy is unchanged until the adapter is trained and given references.

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
CONDITIONS = ("none", "real", "unaligned", "other_task")


def normalize_instruction(text: str) -> str:
    """Task key of an instruction: lowercase, typographic apostrophes folded, whitespace collapsed."""
    return " ".join(text.replace("’", "'").replace("_", " ").strip().lower().split())


def reference_scalars(j: int, similarity: float, align_cost: float, call: int, ref_length: int) -> List[float]:
    """Scalar features of a reference at its step `j` while the current episode is at call `call` (SCALAR_NAMES)."""
    return [j / max(ref_length - 1, 1), similarity, align_cost, call / CALL_SCALE, j / CALL_SCALE,
            ref_length / CALL_SCALE]


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
    """

    def __init__(self, keys: torch.Tensor, lengths: torch.Tensor, vmax: int = 3, gamma: float = 0.01) -> None:
        self.keys = keys.float()  # [P, L, D] unit vectors, zero-padded
        positions = torch.arange(keys.shape[1], device=keys.device)
        self.valid = positions[None] < lengths.to(keys.device)[:, None]
        self.vmax, self.gamma = vmax, gamma
        self.J: Optional[torch.Tensor] = None
        self.calls = 0

    def step(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """query [D] unit key -> per memory: aligned position [P], similarity there [P], mean alignment cost [P]."""
        sims = torch.matmul(self.keys, query.float().to(self.keys.device)).clamp(-1.0, 1.0)  # [P, L]
        cost = (1.0 - sims).masked_fill(~self.valid, float("inf"))
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
        return ends, sims.gather(1, ends[:, None])[:, 0], score


def align_sequence(query_keys: torch.Tensor, keys: torch.Tensor, lengths: torch.Tensor, vmax: int, gamma: float):
    """Offline PMA, call by call as at deployment: query_keys [T, D] -> ends, similarities, mean costs, each [T, P]."""
    pma = ProgressiveAlignment(keys, lengths, vmax, gamma)
    steps = [pma.step(q) for q in query_keys]
    return tuple(torch.stack(x) for x in zip(*steps))


# === Adapter ===
class GatedCrossAttention(nn.Module):
    """
    Action tokens read the reference tokens; `residual` is what the host adds to its hidden states. The output
    projection is zero-initialized, so the residual is exactly 0 at initialization while every weight of the
    projection learns at full speed from the first step (a scalar tanh gate would hold the others' gradients near 0).
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.o_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def residual(self, x, tokens, attend, active):
        """x [B, N, D]; tokens [B, M, D]; attend [B, M] bool (True = may be attended); active [B] -> [B, N, D]."""
        x = x.to(self.q_proj.weight.dtype)
        B, N, D = x.shape
        M, H = tokens.shape[1], self.num_heads
        q = self.q_proj(self.norm(x)).view(B, N, H, D // H).transpose(1, 2)
        k, v = self.kv_proj(tokens).view(B, M, 2, H, D // H).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=attend[:, None, None, :])
        out = self.o_proj(out.transpose(1, 2).reshape(B, N, D))
        return active.to(out.dtype)[:, None, None] * out  # samples without references: exactly 0

    def gate_value(self) -> float:
        """Gain of the zero-initialized projection (RMS weight x sqrt(fan-in))."""
        w = self.o_proj.weight.detach().float()
        return float(w.pow(2).mean().sqrt() * math.sqrt(w.shape[1]))


class ExperienceAdapter(nn.Module):
    """
    Encodes M references into tokens and injects them into the host's action network: cross-attention after the
    blocks listed in `inject_layers`, and (input_inject) the reference chunks, weighted by a softmax over a learned
    score of their scalars, added to action token k's input through a zero-initialized projection of step k, so that
    copying a reference is a linear map from the first block on. The input dict (built by `stack_slots` /
    `ExperienceRuntime` / the training script) holds, for B samples:
        actions [B, M, F, A]  reference action chunk at the matched step (normalized as the host's training data)
        scalars [B, M, S]     `reference_scalars`
        success [B, M]        outcome of the reference episode (1 = success)
        valid   [B, M]        which slots hold a reference
    Every reference contributes F tokens, one per step of its chunk, with a step embedding, so that action token k
    can read step k. A learned null token is always attendable, so a sample with few references never attends to an
    empty set.
    """

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
    ) -> None:
        super().__init__()
        if inject_layers is None:
            inject_layers = [i for i in range(depth) if (i + 1) % inject_every == 0]
        self.inject_layers = sorted(int(i) for i in inject_layers)
        assert all(0 <= i < depth for i in self.inject_layers), f"inject_layers {self.inject_layers} vs depth {depth}"
        self.config = dict(
            hidden_size=hidden_size, num_heads=num_heads, depth=depth, inject_every=inject_every,
            inject_layers=self.inject_layers, action_dim=action_dim, chunk_len=chunk_len, max_refs=max_refs,
            input_inject=input_inject,
        )
        self.step_mlp = nn.Sequential(nn.Linear(action_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.step_emb = nn.Parameter(torch.randn(chunk_len, hidden_size) * 0.02)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(len(SCALAR_NAMES), hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        )
        self.success_emb = nn.Embedding(2, hidden_size)
        self.slot_emb = nn.Parameter(torch.randn(max_refs, hidden_size) * 0.02)
        self.null_token = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.layers = nn.ModuleDict({str(i): GatedCrossAttention(hidden_size, num_heads) for i in self.inject_layers})
        if input_inject:
            self.inject_score = nn.Linear(len(SCALAR_NAMES), 1)  # zero: references weighted uniformly at first
            nn.init.zeros_(self.inject_score.weight)
            nn.init.zeros_(self.inject_score.bias)
            self.inject_proj = nn.Linear(action_dim, hidden_size)  # zero: no effect at initialization
            nn.init.zeros_(self.inject_proj.weight)
            nn.init.zeros_(self.inject_proj.bias)
        else:
            self.inject_score = self.inject_proj = None

    def encode(self, exp: Dict[str, torch.Tensor]):
        """-> (tokens [B, 1 + M*F, H], attend [B, 1 + M*F], active [B], input residual [B, F, H] or None)."""
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
        residual = None
        if self.inject_proj is not None:
            score = self.inject_score(exp["scalars"].to(dtype))[..., 0].masked_fill(~valid, float("-inf"))
            weights = torch.nan_to_num(torch.softmax(score, dim=1)) * active[:, None]  # rows without refs: 0
            chunk = (weights[..., None, None] * actions).sum(dim=1)  # [B, F, A]
            residual = self.inject_proj(chunk)
        return tokens, attend, active, residual

    def layer_residual(self, layer_index: int, x: torch.Tensor, encoded) -> Optional[torch.Tensor]:
        """Residual to add to the output of host block `layer_index` (None if no layer is inserted there)."""
        layer = self.layers[str(layer_index)] if str(layer_index) in self.layers else None
        if layer is None:
            return None
        tokens, attend, active = encoded[:3]
        return layer.residual(x, tokens, attend, active)

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
) -> dict:
    assert keys.shape[0] == actions.shape[0], f"{keys.shape[0]} keys vs {actions.shape[0]} action chunks"
    return {
        "name": name, "task_key": task_key, "keys": keys.half().cpu(), "actions": actions.float().cpu(),
        "sig": None if sig is None else sig.half().cpu(), "success": bool(success), "source": source,
        "meta": dict(meta or {}),
    }


class ExperienceStore:
    """Episodes grouped by task key. Episodes are never modified once added, so stores can share them."""

    def __init__(self, episodes: Optional[Sequence[dict]] = None, info: Optional[dict] = None) -> None:
        self.episodes: List[dict] = []
        self.by_task: Dict[str, List[int]] = {}
        self.info = dict(info or {})  # how the keys were computed (checked against the host at attach time)
        for episode in episodes or []:
            self.add(episode)

    def add(self, episode: dict) -> int:
        index = len(self.episodes)
        self.episodes.append(episode)
        self.by_task.setdefault(episode["task_key"], []).append(index)
        return index

    def copy(self) -> "ExperienceStore":
        return ExperienceStore(self.episodes, self.info)

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

    def save(self, path) -> None:
        torch.save({"version": 2, "info": self.info, "episodes": self.episodes}, path)

    @classmethod
    def load(cls, paths: Sequence) -> "ExperienceStore":
        if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__"):
            paths = [paths]
        store = cls()
        for path in paths:
            data = torch.load(path, map_location="cpu", weights_only=False)
            info = data.get("info", {})
            if store.info and info and info != store.info:
                raise ValueError(f"{path}: key info {info} differs from {store.info}")
            store.info = store.info or dict(info)
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
def reference_slot(ref: dict, j: int, similarity: float, align_cost: float,
                   call: int) -> Tuple[torch.Tensor, List[float], int]:
    T = ref["actions"].shape[0]
    return ref["actions"][j], reference_scalars(j, similarity, align_cost, call, T), int(ref["success"])


def stack_slots(slots: List[Tuple[torch.Tensor, List[float], int]], max_refs: int, chunk_shape) -> Dict[str, torch.Tensor]:
    """One sample's references -> padded [1, max_refs, ...] tensors (CPU)."""
    actions = torch.zeros(1, max_refs, *chunk_shape)
    scalars = torch.zeros(1, max_refs, len(SCALAR_NAMES))
    success = torch.zeros(1, max_refs, dtype=torch.long)
    valid = torch.zeros(1, max_refs, dtype=torch.bool)
    for m, (chunk, scal, succ) in enumerate(slots[:max_refs]):
        actions[0, m], scalars[0, m], success[0, m], valid[0, m] = chunk, torch.tensor(scal), succ, True
    return {"actions": actions, "scalars": scalars, "success": success, "valid": valid}


def empty_slots(batch_size: int, max_refs: int, chunk_shape) -> Dict[str, torch.Tensor]:
    out = stack_slots([], max_refs, chunk_shape)
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
    narrowed to the `pool_size` most similar first frames when there are more); at every call it advances PMA over
    the whole pool and returns the `num_refs` best-aligned candidates as adapter inputs. It also records the episode
    (keys, predicted chunks, signature) so it can be added to a store at the end.
    Conditions: `real`; `unaligned` (control: a random step of each chosen reference); `other_task` (control:
    `num_refs` random successful episodes of other tasks, aligned the same way); `none`.
    """

    def __init__(
        self,
        store: ExperienceStore,
        task_key: str,
        condition: str = "real",
        num_refs: int = 4,
        vmax: int = 3,
        gamma: float = 0.01,
        pool_size: int = 64,
        select: str = "similar",
        sources=("demo", "rollout"),
        success_only: bool = True,
        max_refs: int = 8,
        chunk_shape=(50, 7),
        seed: int = 0,
        device="cuda",
    ) -> None:
        assert condition in CONDITIONS, f"Unknown condition {condition}"
        self.store, self.task_key, self.condition = store, task_key, condition
        self.num_refs, self.vmax, self.gamma = min(num_refs, max_refs), vmax, gamma
        self.pool_size, self.select, self.sources = pool_size, select, tuple(sources)
        self.success_only = success_only
        self.max_refs, self.chunk_shape, self.device = max_refs, tuple(chunk_shape), device
        self.rng = np.random.default_rng(seed)
        self.pool: List[dict] = []
        self.pma: Optional[ProgressiveAlignment] = None
        self.call = 0
        self.keys: List[torch.Tensor] = []
        self.predicted: List[np.ndarray] = []
        self.sig: Optional[torch.Tensor] = None
        self.trace: List[List[Tuple[int, int, float, float]]] = []  # per call: (pool index, j, similarity, cost)

    def _build_pool(self) -> None:
        filters = dict(sources=self.sources, success_only=self.success_only)
        if self.condition == "other_task":
            others = [i for task in sorted(self.store.by_task) if task != self.task_key
                      for i in self.store.candidates(task, **filters)]
            picked = [others[i] for i in self.rng.permutation(len(others))[: self.num_refs]]
        else:
            k = self.pool_size if self.pool_size > 0 else len(self.store)
            picked = self.store.select(self.task_key, k, self.select, sig=self.sig, rng=self.rng, **filters)
        self.pool = [self.store.episodes[i] for i in picked]
        if self.pool:
            keys, lengths = pad_keys([ep["keys"] for ep in self.pool])
            self.pma = ProgressiveAlignment(keys.to(self.device), lengths, self.vmax, self.gamma)

    def __call__(self, key: torch.Tensor, sig: Optional[torch.Tensor] = None) -> Optional[Dict[str, torch.Tensor]]:
        """key [D] unit vector of this call; sig [Ds] unit signature (used on the first call) -> adapter inputs."""
        key = key.float().flatten()
        if self.call == 0:
            self.sig = None if sig is None else sig.float().flatten().half().cpu()
            if self.condition != "none":
                self._build_pool()
        self.keys.append(key.half().cpu())
        exp = None
        if self.pma is not None:
            ends, sims, costs = (x.cpu() for x in self.pma.step(key))
            slots, step_trace = [], []
            for p in torch.argsort(costs)[: self.num_refs].tolist():
                ref, j, sim, cost = self.pool[p], int(ends[p]), float(sims[p]), float(costs[p])
                if self.condition == "unaligned":  # control: a random step of the same reference
                    j = int(self.rng.integers(ref["keys"].shape[0]))
                    sim = float(ref["keys"][j].float() @ key.cpu())
                slots.append(reference_slot(ref, j, sim, cost, self.call))
                step_trace.append((p, j, sim, cost))
            self.trace.append(step_trace)
            exp = stack_slots(slots, self.max_refs, self.chunk_shape)
        self.call += 1
        return exp

    def record_prediction(self, normalized_chunk: np.ndarray) -> None:
        self.predicted.append(np.asarray(normalized_chunk, dtype=np.float32).copy())

    def info(self) -> dict:
        out = {"pool_size": len(self.pool), "num_calls": self.call, "condition": self.condition}
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
        return make_episode(
            self.task_key, torch.stack(self.keys[:n]), torch.from_numpy(chunks), self.sig, success, "rollout",
            name or f"rollout_{time.time_ns()}", meta,
        )
