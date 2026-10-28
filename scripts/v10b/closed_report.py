"""
Report of the V10-1b closed-loop generalization test against the registered criteria (10-24). Needs the open-loop
replay of the SAME sources the closed loop used (runs/v10b/noise_sources; no policy, so GPU-independent) -- computed
here once and cached -- and the true-d upper bound with those sources (v10_oracle_noise.py, error 0).
  1. above replay   : V10-1b - replay, 95% CI clustered by target variant, lower bound > 0;
  2. above base     : V10-1b - base (mean of the two 4080 SUPER seeds) >= +5 pp and lower bound > 0;
  3. little damage  : on the cells where base seed 1 succeeded, V10-1b's failures <= 10%.
    python closed_report.py --workers 12
"""
import argparse
import collections
import json
import multiprocessing as mp
import os

import numpy as np

import v10_oracle_shift2 as O

R = "/root/pace/runs"
RUN = f"{R}/v10b/closed/run1"


def replay_job(a):
    tid, k, seed, src = a
    ok, _ = O.run_episode(tid, k, seed, src["actions"], "replay")
    return dict(target=tid, init=k, source=src["task_id"], success=ok)


def boot(cells, f, groups, n=10000, seed=0):
    """Point estimate and 95% interval of mean_c f(c), resampling the groups."""
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    est = np.mean([f(c) for g in keys for c in groups[g]])
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        vals.append(np.mean([f(c) for i in pick for c in groups[keys[i]]]))
    return est, np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    eps = [json.loads(l) for l in open(f"{RUN}/episodes.jsonl")]
    rp_path = f"{RUN}/replay_same_sources.json"
    if not os.path.exists(rp_path):
        rows = [json.loads(l) for l in open(f"{R}/v10b/noise_sources/episodes.jsonl")]
        srcs = {r["task_id"]: dict(task_id=r["task_id"], actions=np.load(f"{R}/v10b/noise_sources/{r['actions']}").astype(np.float64))
                for r in rows if r["success"]}
        jobs = [(e["task_id"], e["init"], e["seed"], srcs[e["source"]]) for e in eps if e["mode"] == "v10"]
        with mp.get_context("spawn").Pool(args.workers, initializer=O.worker_init) as pool:
            rp = pool.map(replay_job, jobs)
        json.dump(rp, open(rp_path, "w"))
    rp = json.load(open(rp_path))
    ub = json.load(open(f"{R}/rectest/v10_shift/oracle_noise.json"))
    names = {r["task_id"]: r["name"].split("_view_")[0] for r in map(json.loads, open(f"{R}/rectest/v10_shift/base_targets/episodes.jsonl"))}

    cell = collections.defaultdict(lambda: dict(v10=[], base=[], seed1=None, replay=[], upper=[]))
    for e in eps:
        c = cell[(e["task_id"], e["init"])]
        c[e["mode"]].append(e)
        if e["mode"] == "base" and e["seed"] == 1000 + 7 * e["task_id"] + e["init"]:
            c["seed1"] = e["success"]
    for r in rp:
        cell[(r["target"], r["init"])]["replay"].append(r["success"])
    for r in ub:
        cell[(r["target"], r["init"])]["upper"].append(r["err0"]["success"])
    keys = sorted(cell)
    m = lambda c, k: float(np.mean([x["success"] for x in c[k]])) if k in ("v10", "base") else float(np.mean(c[k]))
    by_var, by_task = collections.defaultdict(list), collections.defaultdict(list)
    for k in keys:
        by_var[k[0]].append(cell[k])
        by_task[names[k[0]]].append(cell[k])

    print(f"[cells] {len(keys)}; V10-1b episodes {sum(len(cell[k]['v10']) for k in keys)}, base {sum(len(cell[k]['base']) for k in keys)}")
    for nm, f in (("base (2 seeds, 4080S)", lambda c: m(c, "base")), ("replay (open loop)", lambda c: m(c, "replay")),
                  ("V10-1b", lambda c: m(c, "v10")), ("true-d upper bound (same sources)", lambda c: m(c, "upper"))):
        print(f"  {nm:36s} {100 * np.mean([f(cell[k]) for k in keys]):5.1f}%")
    print("\n[criteria] point estimate [95% CI] in pp; clustered by target variant | by base task")
    for nm, f in (("1. V10-1b - replay", lambda c: m(c, "v10") - m(c, "replay")),
                  ("2. V10-1b - base", lambda c: m(c, "v10") - m(c, "base"))):
        a, b = boot(keys, f, by_var), boot(keys, f, by_task)
        print(f"  {nm:22s} {100 * a[0]:+5.1f} [{100 * a[1]:+5.1f}, {100 * a[2]:+5.1f}] | [{100 * b[1]:+5.1f}, {100 * b[2]:+5.1f}]")
    ok1 = [k for k in keys if cell[k]["seed1"]]
    dmg = 1 - np.mean([m(cell[k], "v10") for k in ok1])
    print(f"  3. damage on the {len(ok1)} cells base seed 1 solved: {100 * dmg:.1f}% of V10-1b episodes fail")
    bad = [k for k in keys if not cell[k]["seed1"]]
    print(f"     rescue on the {len(bad)} cells base seed 1 failed: {100 * np.mean([m(cell[k], 'v10') for k in bad]):.1f}% succeed")

    v = [x for k in keys for x in cell[k]["v10"]]
    fol, nf = [x for x in v if x["follow"]], [x for x in v if not x["follow"]]
    bm = {k: m(cell[k], "base") for k in keys}
    print(f"\n[decision] followed {len(fol)}/{len(v)} ({100 * len(fol) / len(v):.0f}%): success {100 * np.mean([x['success'] for x in fol]):.1f}% "
          f"(base on the same cells {100 * np.mean([bm[(x['task_id'], x['init'])] for x in fol]):.1f}%)")
    print(f"           not followed {len(nf)}: success {100 * np.mean([x['success'] for x in nf]):.1f}% "
          f"(base on the same cells {100 * np.mean([bm[(x['task_id'], x['init'])] for x in nf]):.1f}%)")
    print("\n[per base task] cells | base | replay | V10-1b | upper | followed")
    for t in sorted(by_task):
        cs = by_task[t]
        print(f"  {t[:70]:70s} {len(cs):3d} | {100 * np.mean([m(c, 'base') for c in cs]):5.1f} | "
              f"{100 * np.mean([m(c, 'replay') for c in cs]):5.1f} | {100 * np.mean([m(c, 'v10') for c in cs]):5.1f} | "
              f"{100 * np.mean([m(c, 'upper') for c in cs]):5.1f} | "
              f"{100 * np.mean([bool(x['follow']) for c in cs for x in c['v10']]):5.1f}")


if __name__ == "__main__":
    main()
