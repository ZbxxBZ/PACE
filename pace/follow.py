"""
V10-0 (10-22): a non-learned follower at the policy's output. The base policy runs on every call as usual (its noise
draws are unchanged); for a batch row that follows a reference, the chunk it returns is replaced by the reference's
executed chunk at the matched call:
  - reference (call 0): among the store's successful rollouts told the same instruction (the stored task key: what
    the policy was told, as the v3 retrieval keys them), those whose robot state at call 0 equals this episode's
    (max |diff| over the 8 dims in env units < `setup_tol`) and whose call-0 wrist-camera image tokens are close
    (mean cell cosine >= `wrist_cos`): the same physical setup. The robot state alone is not enough: LIBERO-plus's
    objects-layout variants of one task share their call-0 robot state but not their layout (10-22,
    scripts/v10/v10_layout_match.py); the wrist threshold 0.999 was set on the V7 training collection (variants
    disjoint from the evaluation's; v10_pair_cos_train.py: 8% of its layout pairs pass, 71% of the camera pairs).
    An exact scene match (both cameras' cosine >= `exact_cos`: an earlier success of this very variant) is preferred,
    then the shortest (fewest calls), ties to the earliest stored. No category label;
  - failure memory: a reference followed in a scene (exact match) whose episode failed is not followed there again;
  - `cross_only`: exact matches are never followed (only other variants' successes: the cross-perturbation transfer
    alone, without replays of the variant's own earlier successes);
  - alignment (calls 1, 2, ...): the reference call j' in (j, j + `window`] whose end-effector position is nearest to
    the current one (metres), j the call followed before. If that is more than `max_dist` away, or the reference has
    no later call, the row stops following and runs the base policy for the rest of the episode.
The store keeps what the rows executed (this wraps the policy below PACE's own wrapper, which records the chunks).
States arrive normalized (pi0.5's MEAN_STD preprocessing); they are mapped back to env units with the preprocessor's
stats. Image tokens are PACE's (v2 encoder input, 16 cells per camera, agentview then wrist): the call's own come from
its runtime, the stored episodes' call-0 tokens are kept here (a none-condition run drops the store's image tokens).
Each episode's info gets "follow": reference name / task id / kind, calls followed, why it stopped.
"""
from typing import List, Optional

import numpy as np
import torch

from .pi05 import batch_state


def state_stats(preprocessor):
    """Mean and std of observation.state from the policy preprocessor's normalizer step."""
    for step in getattr(preprocessor, "steps", []):
        stats = getattr(step, "stats", None)
        if stats and "observation.state" in stats:
            s = stats["observation.state"]
            return (np.asarray(torch.as_tensor(s["mean"]).float().cpu()),
                    np.asarray(torch.as_tensor(s["std"]).float().cpu()))
    raise ValueError("no observation.state stats in the preprocessor")


def cams_cos(a: torch.Tensor, b: torch.Tensor) -> List[float]:
    """Mean cell cosine per camera of two calls' image tokens [2 * 16, D] (agentview, wrist)."""
    a, b = a.float().view(2, 16, -1), b.float().view(2, 16, -1)
    return [float(torch.nn.functional.cosine_similarity(a[c], b[c], dim=-1).mean()) for c in range(2)]


class _Row:
    def __init__(self, task_key: str):
        self.task_key = task_key
        self.call = 0
        self.ref: Optional[dict] = None
        self.ref_state: Optional[np.ndarray] = None  # [T, Ds] env units
        self.j = -1
        self.status = "new"  # new -> follow -> lost / ended; or base (no reference)
        self.followed = 0
        self.stopped_at = None
        self.dists: List[float] = []
        self.kind, self.wrist, self.candidates = None, None, 0

    def info(self) -> dict:
        out = {"status": self.status, "followed_calls": self.followed, "candidates": self.candidates}
        if self.ref is not None:
            out.update(ref=self.ref["name"], ref_kind=self.kind, ref_wrist_cos=self.wrist,
                       ref_task_id=self.ref["meta"].get("task_id"), ref_round=self.ref["meta"].get("round"),
                       ref_calls=int(self.ref["actions"].shape[0]), stopped_at=self.stopped_at,
                       max_dist=float(max(self.dists)) if self.dists else 0.0)
        return out


class Follower:
    def __init__(self, policy, preprocessor, max_dist: float = 0.02, window: int = 3, setup_tol: float = 1e-3,
                 wrist_cos: float = 0.999, exact_cos: float = 0.99999, cross_only: bool = False):
        self.mean, self.std = state_stats(preprocessor)
        self.max_dist, self.window, self.setup_tol = float(max_dist), int(window), float(setup_tol)
        self.wrist_cos, self.exact_cos, self.cross_only = float(wrist_cos), float(exact_cos), bool(cross_only)
        self.pace = None
        self.rows: Optional[List[_Row]] = None
        self.runtimes = None
        self.vis0 = {}  # stored episode name -> its call-0 image tokens
        self.failed = []  # (call-0 image tokens of a failed episode, name of the reference it followed)
        original = policy.predict_action_chunk

        def predict_action_chunk(batch, **kwargs):
            return self.follow(batch, original(batch, **kwargs))

        # must be in place before PACE attaches (it wraps whatever predict_action_chunk is then)
        policy.predict_action_chunk = predict_action_chunk

    def bind(self, pace) -> None:
        """Start / close the rows with PACE's episodes (its begin / end)."""
        assert pace.needs_state, "the follower needs the stored rollouts' robot states (v3 retrieval or features)"
        assert pace.features, "the follower compares call-0 image tokens: attach an adapter with the v2 encoder"
        self.pace = pace
        begin, end = pace.begin, pace.end

        def begin_(*a, **k):
            begin(*a, **k)
            self.runtimes = list(pace.runtimes)
            self.rows = [_Row(r.task_key) for r in self.runtimes]

        def end_(successes, *a, **k):
            rows, self.rows = self.rows, None
            vis0 = [r.vis[0] if r.vis else None for r in self.runtimes]
            infos = end(successes, *a, **k)
            for b, (info, row) in enumerate(zip(infos, rows or [])):
                info["follow"] = row.info()
                if "stored_as" in info and vis0[b] is not None:
                    self.vis0[info["stored_as"]] = vis0[b]
                if row.ref is not None and not successes[b] and vis0[b] is not None:
                    self.failed.append((vis0[b], row.ref["name"]))
            self.runtimes = None
            return infos

        pace.begin, pace.end = begin_, end_

    def env_units(self, state) -> np.ndarray:
        return np.asarray(torch.as_tensor(state).float().cpu()) * self.std + self.mean

    def pick(self, task_key: str, s0: np.ndarray, vis0: torch.Tensor):
        """(reference episode, "exact" / "cross", its wrist cosine), or Nones; and how many passed the state check."""
        store = self.pace.store if self.pace is not None else None
        if store is None:
            return (None, None, None), 0
        best, best_rank, n = None, None, 0
        for ep in store.episodes:
            if (ep["source"] != "rollout" or not ep["success"] or ep["task_key"] != task_key or "state" not in ep
                    or ep["state"].shape[0] == 0):
                continue
            if float(np.abs(self.env_units(ep["state"][0]) - s0).max()) >= self.setup_tol:
                continue
            ref_vis0 = self.vis0.get(ep["name"], ep["vis"][0] if "vis" in ep else None)
            if ref_vis0 is None:
                continue
            n += 1
            agent, wrist = cams_cos(vis0, ref_vis0)
            if wrist < self.wrist_cos:
                continue
            if any(name == ep["name"] and min(cams_cos(vis0, v)) >= self.exact_cos for v, name in self.failed):
                continue
            exact = min(agent, wrist) >= self.exact_cos
            if exact and self.cross_only:
                continue
            rank = (not exact, int(ep["actions"].shape[0]))
            if best is None or rank < best_rank:
                best, best_rank = (ep, "exact" if exact else "cross", wrist), rank
        return (best if best is not None else (None, None, None)), n

    def step(self, row: _Row, s: np.ndarray, vis0: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        call = row.call
        row.call += 1
        if row.status == "new":
            (ref, kind, wrist), row.candidates = self.pick(row.task_key, s, vis0)
            if ref is None:
                row.status = "base"
                return None
            row.ref, row.ref_state, row.j, row.status = ref, self.env_units(ref["state"]), 0, "follow"
            row.kind, row.wrist, row.followed, row.dists = kind, wrist, 1, [0.0]
            return ref["actions"][0]
        if row.status != "follow":
            return None
        T = row.ref_state.shape[0]
        cand = list(range(row.j + 1, min(row.j + 1 + self.window, T)))
        if not cand:
            row.status, row.stopped_at = "ended", call
            return None
        d = np.linalg.norm(row.ref_state[cand, :3] - s[None, :3], axis=1)
        k = int(np.argmin(d))
        row.dists.append(float(d[k]))
        if d[k] > self.max_dist:
            row.status, row.stopped_at = "lost", call
            return None
        row.j = cand[k]
        row.followed += 1
        return row.ref["actions"][row.j]

    def follow(self, batch, actions: torch.Tensor) -> torch.Tensor:
        if self.rows is None:
            return actions
        states = self.env_units(batch_state(batch))
        assert states.shape[0] == len(self.rows), f"batch of {states.shape[0]} vs {len(self.rows)} rows"
        out = None
        for b, row in enumerate(self.rows):
            # call 0: this call's image tokens (PACE's runtime recorded them while the base policy ran)
            vis0 = self.runtimes[b].vis[0] if row.status == "new" else None
            chunk = self.step(row, states[b], vis0)
            if chunk is None:
                continue
            if out is None:
                out = actions.clone()
            n = min(chunk.shape[0], out.shape[1])
            assert chunk.shape[-1] == out.shape[-1], f"reference actions {tuple(chunk.shape)} vs {tuple(out.shape)}"
            out[b, :n] = chunk[:n].to(device=out.device, dtype=out.dtype)
        return actions if out is None else out
