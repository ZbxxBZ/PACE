"""
Offline comparison of the classic and v3 retrievals (no policy rollouts, no GPU needed): held-out library demos are
replayed call by call through `core.ExperienceRuntime` exactly as deployed, against a store of the other demos, and
the references it returns are scored.

Per task, --query_frac of the demos are queries and the rest the store. Every setting is run at every --key_noise
level: the queries' image keys and first-frame signatures get Gaussian noise of that norm (then re-normalized), a crude
stand-in for a changed camera, lighting or image noise (robot states stay clean, as they would). Settings:
  classic                      the query's own task looked up by its exact instruction (what v1 / v2 use)
  v3 sw=<state_weight> tw=<text_weight> text=<shown|none>
                               candidates from the tasks closest in instruction (shown: the query's own instruction;
                               none: no instruction, scene only) and first frame, state-weighted alignment
Scores (per call, averaged over calls; refs = the --num_refs references the runtime returns):
  same1 / same4   fraction of the top reference / of the refs that come from the query's own task
  prog1           |progress of the top reference's matched step - progress of the query call| (same-task top only)
  act1 / act4     MSE of the first --exec_steps actions of the top reference's chunk / of the refs' mean chunk against
                  the query's own chunk (what the adapter reads; lower = more useful)
  cost1 / scost1  image cost (1 - cos) and state cost of the top reference

    python -m pace.retrieval_study --library runs/libs/libero4_grid_pca128_feat.pt \
        --policy_path lerobot/pi05_libero_finetuned --out runs/libs/retrieval_study.json
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .core import SCALAR_NAMES, V3_SCALAR_NAMES, ExperienceRuntime, ExperienceStore, has_states
from .pi05 import InstructionText


def noisy(x: torch.Tensor, level: float, gen: torch.Generator) -> torch.Tensor:
    """Unit vectors x [..., d] plus Gaussian noise of norm ~level, re-normalized."""
    if level <= 0:
        return x.float()
    noise = torch.randn(x.shape, generator=gen) / x.shape[-1] ** 0.5
    return F.normalize(x.float() + level * noise, dim=-1)


def run_query(store, query, setting, args, text, noise, gen):
    """One held-out demo through a runtime -> per-call score rows."""
    v3 = setting["retrieval"] == "v3"
    kwargs = dict(retrieval=setting["retrieval"], n_scalars=len(V3_SCALAR_NAMES if v3 else SCALAR_NAMES))
    if v3:
        kwargs.update(state_weight=setting["state_weight"], state_sigma=args.state_sigma,
                      text_weight=setting["text_weight"], text_tasks=args.text_tasks, scene_tasks=args.scene_tasks,
                      task_text=text.similarities, query_text=query["task_key"] if setting["text"] == "shown" else None)
    runtime = ExperienceRuntime(store, query["task_key"], condition="real", num_refs=args.num_refs, vmax=args.vmax,
                                gamma=args.gamma, align=args.align, pool_size=args.pool_size, device=args.device,
                                true_task=query["task_key"], **kwargs)
    keys = noisy(query["keys"], noise, gen)
    sig = noisy(query["sig"], noise, gen)
    T = keys.shape[0]
    rows = []
    for t in range(T):
        exp = runtime(keys[t], sig, None, query["state"][t])
        if exp is None or not runtime.trace or not runtime.trace[-1]:
            continue
        refs = runtime.trace[-1]
        truth = query["actions"][t][: args.exec_steps]
        chunks = [runtime.pool[p]["actions"][j][: args.exec_steps] for p, j, _, _ in refs]
        p0, j0, sim0, _ = refs[0]
        top = runtime.pool[p0]
        same = [runtime.pool[p]["task_key"] == query["task_key"] for p, *_ in refs]
        row = {"same1": float(same[0]), "same4": float(np.mean(same)),
               "act1": float(((chunks[0] - truth) ** 2).mean()),
               "act4": float(((torch.stack(chunks).mean(dim=0) - truth) ** 2).mean()),
               "cost1": 1.0 - sim0,
               "scost1": float(exp["scalars"][0, 0, -1]) if v3 else float("nan")}
        if same[0]:
            row["prog1"] = abs(j0 / max(top["keys"].shape[0] - 1, 1) - t / max(T - 1, 1))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="Demo library with robot states (pace.build_library)")
    parser.add_argument("--policy_path", default="lerobot/pi05_libero_finetuned",
                        help="Checkpoint whose input token embeddings compare instructions (read on the CPU)")
    parser.add_argument("--tokenizer", default="google/paligemma-3b-pt-224")
    parser.add_argument("--query_frac", type=float, default=0.1)
    parser.add_argument("--max_queries", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--key_noise", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--state_weights", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    parser.add_argument("--text_weights", type=float, nargs="+", default=[0.1, 0.2, 0.4])
    parser.add_argument("--text_modes", nargs="+", default=["shown", "none"], choices=("shown", "none"))
    parser.add_argument("--state_sigma", type=float, default=0.3)
    parser.add_argument("--text_tasks", type=int, default=3)
    parser.add_argument("--scene_tasks", type=int, default=2)
    parser.add_argument("--num_refs", type=int, default=4)
    parser.add_argument("--pool_size", type=int, default=64)
    parser.add_argument("--vmax", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--align", choices=("pma", "nn"), default="pma")
    parser.add_argument("--exec_steps", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    library = ExperienceStore.load(args.library)
    demos = [ep for ep in library.episodes if ep["source"] == "demo"]
    assert all(has_states(ep) for ep in demos), f"{args.library} has no robot states: rebuild it with build_library"
    rng = np.random.default_rng(args.seed)
    by_task = collections.defaultdict(list)
    for ep in demos:
        by_task[ep["task_key"]].append(ep)
    queries, pool = [], []
    for task in sorted(by_task):
        eps = [by_task[task][i] for i in rng.permutation(len(by_task[task]))]
        n_q = max(1, int(round(len(eps) * args.query_frac)))
        queries += eps[:n_q]
        pool += eps[n_q:]
    if args.max_queries:
        queries = [queries[i] for i in sorted(rng.permutation(len(queries))[: args.max_queries])]
    store = ExperienceStore(pool, library.info, library.key_transform)
    text = InstructionText.from_checkpoint(args.policy_path, args.tokenizer, sorted(store.by_task))
    tasks = sorted(store.by_task)
    embedded = torch.stack([text.embed(t) for t in tasks])
    others = (embedded @ embedded.T).masked_fill(torch.eye(len(tasks), dtype=torch.bool), float("nan"))
    print(f"[study] {len(queries)} query demos, {len(pool)} store demos, {len(tasks)} tasks; instruction similarity "
          f"between different tasks: median {others.nanmedian():.3f}, max {others.nan_to_num(-1).max():.3f}",
          flush=True)

    settings = [{"name": "classic", "retrieval": "classic"}]
    for mode in args.text_modes:
        for sw in args.state_weights:
            for tw in (args.text_weights if mode == "shown" else [0.0]):
                settings.append({"name": f"v3 sw={sw:g} tw={tw:g} text={mode}", "retrieval": "v3",
                                 "state_weight": sw, "text_weight": tw, "text": mode})
    metrics = ("same1", "same4", "prog1", "act1", "act4", "cost1", "scost1")
    print(f"{'setting':34s} {'noise':>5s} " + " ".join(f"{m:>7s}" for m in metrics), flush=True)
    results = []
    for noise in args.key_noise:
        for setting in settings:
            started = time.time()
            rows = []
            for q, query in enumerate(queries):
                gen = torch.Generator().manual_seed(args.seed * 100003 + q)  # the same noise in every setting
                rows += run_query(store, query, setting, args, text, noise, gen)
            summary = {m: float(np.nanmean([r[m] for r in rows if m in r])) if any(m in r for r in rows)
                       else float("nan") for m in metrics}
            results.append({**setting, "noise": noise, "calls": len(rows), "seconds": time.time() - started,
                            **summary})
            print(f"{setting['name']:34s} {noise:5.2f} " + " ".join(f"{summary[m]:7.4f}" for m in metrics),
                  flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=2)
        print(f"[study] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
