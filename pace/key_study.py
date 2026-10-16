"""
Offline comparison of retrieval-key transforms for PACE's alignment (no closed-loop evaluation).

pi0.5's raw keys (image-token means, unit-normalized) are nearly constant within LIBERO (mean PMA similarity ~0.996),
so alignment has little signal. This script compares alternatives on the demos of a library:

  1. --extract (GPU, vision tower only): for every call of every library demo, the raw per-camera image-token mean
     [C, D] and the 4x4 grid average [C, 16, D] (before any normalization), saved with the demos' action chunks.
  2. evaluation (CPU is enough): per task, --query_frac of the demos are queries and the rest the pool. Every key
     variant `<base>:<transform>` is fitted on the pool calls only, the queries are aligned call by call to the pool
     (PMA, as deployed, and frame nearest-neighbour as RTCF's Frame-NN ablation) and scored:
       prog_err   |progress of the matched step - progress of the query call|, top-1 reference (progress = index / (T-1))
       backward   fraction of consecutive calls where the top-1 reference stays the same but its matched step goes back
       act1/act4  MSE of the executed part (first --exec_steps steps) of the top-1 reference chunk / of the mean of the
                  top-4 chunks against the query's own chunk (what the adapter reads; lower = more useful reference)
       same_wins  fraction of queries whose best same-task pool episode beats the best of --max_others random
                  other-task pool episodes (final call)
       gap        (best other-task cost - best same-task cost) / median best per-call cost (final call)
     Reference points for act: `time` (4 random same-task references read at the proportional step), `oracle` (best
     step of the best pool episode per call, an unreachable lower bound) and `pool_mean` (mean chunk of the pool).

  bases       mean       per-camera token means, each unit-normalized, concatenated (cameras --cameras)
              grid       per-camera 4x4 grid cells, each unit-normalized, concatenated
              mean_cam<c> one camera only
  transforms  none (normalize only = the current keys for base `mean`), center, zscore, abtt<d> (center, remove the top
              d principal components), pca<k>, pcaw<k> (PCA + whitening), pls<k> (directions of maximal covariance with
              the executed action chunk), lda<k> (PCA to 256 dims, then LDA on task x progress-bin classes)
  gammas      PMA's soft prior: absolute values, or rel<r> = r x median best per-call cost of that variant

    python -m pace.key_study --extract --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
        --dataset_root <root> --library libs/goal_demo.pt --features libs/goal_key_features.pt
    python -m pace.key_study --features libs/goal_key_features.pt --out libs/goal_key_study.json
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core import ExperienceStore, align_sequence, pad_keys

DEFAULT_VARIANTS = (
    "mean:none", "mean:center", "mean:zscore", "mean:abtt3", "mean:pca32", "mean:pca64", "mean:pca128",
    "mean:pcaw64", "mean:pls32", "mean:lda32", "mean_cam0:pca64", "grid:center", "grid:pca128", "grid:pls32",
)


# === 1. Feature extraction ===
@torch.no_grad()
def extract(args) -> None:
    from torch.utils.data import DataLoader, default_collate

    from .common import load_dataset, load_policy

    policy, preprocessor = load_policy(args.policy_path, "cuda", n_action_steps=args.n_action_steps)
    dataset = load_dataset(policy, args.dataset_repo_id, args.dataset_root)
    store = ExperienceStore.load(args.library)
    episodes = [ep for ep in store.episodes if ep["source"] == "demo"]
    loader = DataLoader(dataset, batch_sampler=[list(ep["meta"]["dataset_indices"]) for ep in episodes],
                        num_workers=args.num_workers, collate_fn=default_collate)
    means, grids, started = [], [], time.time()
    for n, raw in enumerate(loader, 1):
        batch = preprocessor(raw)
        images, img_masks = policy._preprocess_images(batch)
        embs = policy.model._embed_images(images, img_masks)
        m, g = [], []
        for c in args.cameras:
            e = embs[c].float()  # [T, N, D]
            T, N, D = e.shape
            side = int(round(N ** 0.5))
            m.append(e.mean(dim=1))
            g.append(F.adaptive_avg_pool2d(e.transpose(1, 2).reshape(T, D, side, side), 4).flatten(2).transpose(1, 2))
        means.append(torch.stack(m, dim=1).half().cpu())  # [T, C, D]
        grids.append(torch.stack(g, dim=1).half().cpu())  # [T, C, 16, D]
        if n % 50 == 0:
            print(f"[extract] {n}/{len(episodes)} episodes ({time.time() - started:.0f}s)", flush=True)
    torch.save({
        "names": [ep["name"] for ep in episodes], "task": [ep["task_key"] for ep in episodes],
        "actions": [ep["actions"] for ep in episodes], "mean": means, "grid": grids, "cameras": list(args.cameras),
        "library_keys": [ep["keys"] for ep in episodes],
    }, args.features)
    print(f"[extract] saved {len(episodes)} episodes to {args.features}", flush=True)


# === 2. Key variants ===
def base_vectors(feats, base: str):
    """Per-episode [T, d] float32 vectors of a base (before the fitted transform)."""
    out = []
    for m, g in zip(feats["mean"], feats["grid"]):
        if base == "mean":
            v = F.normalize(m.float(), dim=-1).flatten(1)
        elif base.startswith("mean_cam"):
            v = F.normalize(m[:, int(base[8:])].float(), dim=-1)
        elif base == "grid":
            v = F.normalize(g.float(), dim=-1).flatten(1)
        else:
            raise ValueError(f"Unknown base {base}")
        out.append(v)
    return out


def top_components(X: torch.Tensor, k: int):
    """Top-k principal directions [d, k] and variances [k] of centered X [n, d]."""
    q = min(k + 10, min(X.shape))
    U, S, V = torch.pca_lowrank(X, q=q, center=False, niter=4)
    return V[:, :k], (S[:k] ** 2) / max(X.shape[0] - 1, 1)


class Transform:
    """A transform fitted on pool vectors; apply(v [T, d]) -> unit vectors [T, d']."""

    def __init__(self, name: str, X: torch.Tensor, Y: torch.Tensor, classes: torch.Tensor) -> None:
        self.name, self.mu, self.sd, self.W = name, None, None, None
        if name == "none":
            return
        self.mu = X.mean(dim=0)
        Xc = X - self.mu
        if name == "center":
            return
        if name == "zscore":
            self.sd = Xc.std(dim=0).clamp_min(1e-6)
        elif name.startswith("abtt"):
            V, _ = top_components(Xc, int(name[4:]))
            self.W = ("remove", V)
        elif name.startswith("pcaw"):
            V, var = top_components(Xc, int(name[4:]))
            self.W = ("project", V / var.clamp_min(1e-12).sqrt())
        elif name.startswith("pca"):
            V, _ = top_components(Xc, int(name[3:]))
            self.W = ("project", V)
        elif name.startswith("pls"):
            k = int(name[3:])
            Yc = (Y - Y.mean(dim=0)) / Y.std(dim=0).clamp_min(1e-6)
            U, _, _ = torch.linalg.svd(Xc.T @ Yc, full_matrices=False)  # [d, min(d, dy)]
            assert k <= U.shape[1], f"pls{k}: the action target has only {U.shape[1]} dims"
            self.W = ("project", U[:, :k])
        elif name.startswith("lda"):
            from scipy.linalg import eigh

            k = int(name[3:])
            P, _ = top_components(Xc, min(256, Xc.shape[1]))
            Z = (Xc @ P).double()
            Sw = torch.zeros(Z.shape[1], Z.shape[1], dtype=torch.float64)
            Sb = torch.zeros_like(Sw)
            for c in classes.unique():
                Zc = Z[classes == c]
                mc = Zc.mean(dim=0)
                Sw += (Zc - mc).T @ (Zc - mc)
                Sb += len(Zc) * torch.outer(mc, mc)
            Sw += 1e-3 * torch.trace(Sw) / Sw.shape[0] * torch.eye(Sw.shape[0], dtype=torch.float64)
            vals, vecs = eigh(Sb.numpy(), Sw.numpy())
            L = torch.from_numpy(vecs[:, np.argsort(vals)[::-1][:k]].copy()).float()
            self.W = ("project", P @ L)
        else:
            raise ValueError(f"Unknown transform {name}")

    def apply(self, v: torch.Tensor) -> torch.Tensor:
        if self.mu is not None:
            v = v - self.mu
        if self.sd is not None:
            v = v / self.sd
        if self.W is not None:
            kind, W = self.W
            v = v - (v @ W) @ W.T if kind == "remove" else v @ W
        return F.normalize(v, dim=-1)


# === 3. Scoring ===
def chunk_mse(a: torch.Tensor, b: torch.Tensor, steps: int) -> torch.Tensor:
    return ((a[..., :steps, :] - b[..., :steps, :]) ** 2).mean(dim=(-1, -2))


def nn_align(query: torch.Tensor, keys: torch.Tensor, lengths: torch.Tensor):
    """Frame nearest neighbour: per call and pool episode, the most similar step and its cost (no history)."""
    valid = torch.arange(keys.shape[1])[None] < lengths[:, None]
    sims = torch.einsum("td,pld->tpl", query, keys).masked_fill(~valid[None], -2.0)
    best, ends = sims.max(dim=2)
    return ends, best, 1.0 - best


def score_variant(keys, feats, queries, pools, others, gamma: float, method: str, exec_steps: int):
    acts = feats["actions"]
    prog, backward, act1, act4, wins, gaps, best_costs = [], [], [], [], [], [], []
    for q in queries:
        pool = pools[q]
        kp, lp = pad_keys([keys[p] for p in pool])
        if method == "pma":
            ends, sims, costs = align_sequence(keys[q], kp, lp, 3, gamma)
        else:
            ends, sims, costs = nn_align(keys[q], kp, lp)
        Tq = keys[q].shape[0]
        prev = None
        for t in range(Tq):
            order = torch.argsort(costs[t])
            p0, j0 = int(order[0]), int(ends[t, order[0]])
            Tr = keys[pool[p0]].shape[0]
            prog.append(abs(j0 / max(Tr - 1, 1) - t / max(Tq - 1, 1)))
            if prev is not None and prev[0] == p0:
                backward.append(float(j0 < prev[1]))
            prev = (p0, j0)
            truth = acts[q][t]
            act1.append(float(chunk_mse(acts[pool[p0]][j0], truth, exec_steps)))
            top = [acts[pool[int(p)]][int(ends[t, p])] for p in order[:4]]
            act4.append(float(chunk_mse(torch.stack(top).mean(dim=0), truth, exec_steps)))
            best_costs.append(float(costs[t, order[0]]))
        if others[q]:
            ko, lo = pad_keys([keys[o] for o in others[q]])
            if method == "pma":
                _, _, oc = align_sequence(keys[q], ko, lo, 3, gamma)
            else:
                _, _, oc = nn_align(keys[q], ko, lo)
            same, other = float(costs[-1].min()), float(oc[-1].min())
            wins.append(float(same < other))
            gaps.append(other - same)
    med = float(np.median(best_costs)) or 1e-9
    return {
        "prog_err": float(np.mean(prog)), "backward": float(np.mean(backward)) if backward else 0.0,
        "act1": float(np.mean(act1)), "act4": float(np.mean(act4)),
        "same_wins": float(np.mean(wins)) if wins else float("nan"),
        "gap": float(np.mean(gaps)) / med if gaps else float("nan"), "median_cost": med,
    }


def reference_points(feats, queries, pools, exec_steps: int, rng):
    acts = feats["actions"]
    time_err, oracle_err, mean_err = [], [], []
    for q in queries:
        pool = pools[q]
        stacked = torch.cat([acts[p] for p in pool])  # every step of every pool episode
        pool_mean = stacked.mean(dim=0)
        Tq = acts[q].shape[0]
        refs = [pool[i] for i in rng.permutation(len(pool))[:4]]
        for t in range(Tq):
            truth = acts[q][t]
            chunks = [acts[r][round(t / max(Tq - 1, 1) * (acts[r].shape[0] - 1))] for r in refs]
            time_err.append(float(chunk_mse(torch.stack(chunks).mean(dim=0), truth, exec_steps)))
            oracle_err.append(float(chunk_mse(stacked, truth, exec_steps).min()))
            mean_err.append(float(chunk_mse(pool_mean, truth, exec_steps)))
    return {"time": float(np.mean(time_err)), "oracle": float(np.mean(oracle_err)),
            "pool_mean": float(np.mean(mean_err))}


def evaluate(args) -> None:
    feats = torch.load(args.features, map_location="cpu", weights_only=False)
    rng = np.random.default_rng(args.seed)
    by_task = collections.defaultdict(list)
    for i, task in enumerate(feats["task"]):
        by_task[task].append(i)
    queries, train = [], []
    for task, eps in sorted(by_task.items()):
        perm = [eps[i] for i in rng.permutation(len(eps))]
        n_q = max(1, int(round(len(eps) * args.query_frac)))
        queries += perm[:n_q]
        train += perm[n_q:]
    train_set = set(train)
    pools = {q: [p for p in by_task[feats["task"][q]] if p in train_set] for q in queries}
    others = {}
    for q in queries:  # a random subset of the other tasks' pool demos (grid keys are 64k-dimensional)
        cand = [p for p in train if feats["task"][p] != feats["task"][q]]
        others[q] = [cand[i] for i in rng.permutation(len(cand))[: args.max_others]]
    print(f"[study] {len(queries)} query demos, {len(train)} pool demos, {len(by_task)} tasks, "
          f"exec_steps={args.exec_steps}", flush=True)

    # Fitting targets: executed action chunk (pls) and task x progress-bin classes (lda), pool calls only
    Y = torch.cat([feats["actions"][p][:, : args.exec_steps].flatten(1) for p in train]).float()
    task_ids = {t: i for i, t in enumerate(sorted(by_task))}
    classes = torch.cat([
        task_ids[feats["task"][p]] * args.progress_bins
        + torch.clamp((torch.arange(feats["actions"][p].shape[0]) * args.progress_bins) // feats["actions"][p].shape[0],
                      max=args.progress_bins - 1)
        for p in train
    ])
    refs = reference_points(feats, queries, pools, args.exec_steps, np.random.default_rng(args.seed + 1))
    print(f"[study] reference act MSE: time {refs['time']:.4f}  pool_mean {refs['pool_mean']:.4f}  "
          f"oracle {refs['oracle']:.4f}", flush=True)

    results, bases = [], {}
    header = f"{'variant':18s} {'gamma':>8s} {'method':>6s} {'prog_err':>8s} {'backward':>8s} {'act1':>7s} " \
             f"{'act4':>7s} {'same_win':>8s} {'gap':>6s} {'med_cost':>8s}"
    print(header, flush=True)
    for variant in args.variants:
        base, transform = variant.split(":")
        if base not in bases:
            bases[base] = base_vectors(feats, base)
        vectors = bases[base]
        started = time.time()
        fitted = Transform(transform, torch.cat([vectors[p] for p in train]), Y, classes)
        keys = [fitted.apply(v) for v in vectors]
        fit_s = time.time() - started
        for method in args.methods:
            gammas = args.gammas if method == "pma" else ["-"]
            for g in gammas:
                gamma = 0.0
                if method == "pma":
                    if str(g).startswith("rel"):
                        probe = score_variant(keys, feats, queries[:20], pools, {q: [] for q in queries}, 0.0, "nn",
                                              args.exec_steps)
                        gamma = float(g[3:]) * probe["median_cost"]
                    else:
                        gamma = float(g)
                row = score_variant(keys, feats, queries, pools, others, gamma, method, args.exec_steps)
                row.update(variant=variant, method=method, gamma=gamma, gamma_spec=str(g), dim=int(keys[0].shape[1]),
                           fit_s=fit_s)
                results.append(row)
                print(f"{variant:18s} {gamma:8.4f} {method:>6s} {row['prog_err']:8.3f} {row['backward']:8.3f} "
                      f"{row['act1']:7.4f} {row['act4']:7.4f} {row['same_wins']:8.3f} {row['gap']:6.2f} "
                      f"{row['median_cost']:8.4f}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "reference": refs, "results": results}, f, indent=2)
        print(f"[study] wrote {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature file (written by --extract, read otherwise)")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--policy_path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--dataset_repo_id", default="lerobot/libero")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--library", default=None, help="Demo library whose calls are featurized (--extract)")
    parser.add_argument("--n_action_steps", type=int, default=10)
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--methods", nargs="+", default=["pma", "nn"], choices=("pma", "nn"))
    parser.add_argument("--gammas", nargs="+", default=["0.01", "rel0.1"])
    parser.add_argument("--exec_steps", type=int, default=10, help="Executed steps per call scored in act1/act4")
    parser.add_argument("--progress_bins", type=int, default=5, help="Progress bins of the LDA classes")
    parser.add_argument("--query_frac", type=float, default=0.2)
    parser.add_argument("--max_others", type=int, default=60, help="Other-task pool demos per query (same_wins/gap)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.extract:
        assert args.library, "--extract needs --library"
        extract(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
