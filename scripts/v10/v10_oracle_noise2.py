"""
d error that shrinks near the contact (10-23): v10_oracle_noise.py added a constant error over a whole phase, but a
d re-estimated every call is coarse far away and fine near the object. Here the error in a phase points in one random
direction (fixed per (target, init, source, phase)) and its length goes from --far (m) when the reference's hand is
>= --reach away from this phase's contact point (xy, along the reference path) linearly down to --near at the
contact. Same 100 cells, sources and seeds as v10_oracle_noise.py (collect_pi05.py noise_sources).
    python v10_oracle_noise2.py --workers 12 -> runs/rectest/v10_shift/oracle_noise2.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import time

import numpy as np

import v10_oracle_noise as N
import v10_oracle_shift2 as O

OUT = "/root/pace/runs/rectest/v10_shift/oracle_noise2.json"


def tapered_shift(rec, phases, target_start, n, blend, path, far, near, reach, rng):
    dv = N.noisy_shift(rec, phases, target_start, n, blend, 0.0, rng)  # the true d, blended as everywhere else
    for s, e, _ in phases:
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        contact = path[min(max(e - 1, 0), n - 1)][:2]
        for t in range(s, min(max(e, s + 1), n)):
            dist = float(np.linalg.norm(path[t][:2] - contact))
            mag = near + (far - near) * min(1.0, dist / reach)
            dv[t, :2] += mag * u
    return dv


def job(args):
    (tid, k, seed), sources, levels, reach = args
    tstart = O.target_start_positions(tid, k, seed)
    out = []
    for src in sources:
        rec = src["rec"]
        n = len(src["actions"])
        phases = O.phase_targets(rec)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        row = dict(target=tid, init=k, source=src["task_id"])
        for far, near in levels:
            rng = np.random.default_rng([tid, k, src["task_id"], int(far * 1000), int(near * 1000)])
            dvec = tapered_shift(rec, phases, tstart, n, 15, path, far, near, reach, rng)
            ok, steps = O.run_episode(tid, k, seed, src["actions"], "track", path, dvec, 1.0)
            row[f"far{int(far * 1000)}_near{int(near * 1000)}"] = dict(success=ok, steps=steps)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["0.03:0.005", "0.03:0.01", "0.05:0.005", "0.05:0.01"])
    ap.add_argument("--reach", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sources", default="/root/pace/runs/v10b/noise_sources")
    args = ap.parse_args()
    levels = [tuple(float(x) for x in l.split(":")) for l in args.levels]
    rows = [json.loads(l) for l in open(f"{args.sources}/episodes.jsonl")]
    sources = {}
    for r in rows:
        if r["success"]:
            sources[(r["base"], r["task_id"])] = dict(key=(r["base"], r["task_id"]), task_id=int(r["task_id"]),
                                                      init_index=int(r["init"]), seed=int(r["seed"]),
                                                      category=r["category"],
                                                      actions=np.load(f"{args.sources}/{r['actions']}").astype(np.float64))
    targets = [json.loads(l) for l in open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl")]
    cams = sorted({r["task_id"]: dict(r, base=r["name"].split("_view_")[0]) for r in targets}.items())
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=O.worker_init) as pool:
        recs = dict(pool.map(O.rec_job, list(sources.values())))
        for k, r in recs.items():
            sources[k]["rec"] = r
        jobs = []
        for tid, r in cams:
            pool_src = [s for (b, _), s in sources.items() if b == r["base"] and s["rec"]["success"]][:3]
            for k in (1, 2, 3, 4, 5):
                jobs.append(((tid, k, 1000 + 7 * tid + k), pool_src, levels, args.reach))
        keys = [f"far{int(f * 1000)}_near{int(n * 1000)}" for f, n in levels]
        results = []
        for out in pool.imap_unordered(job, jobs):
            results += out
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"[progress] {len(results)} pairs ({time.time() - t0:.0f}s): " + ", ".join(
                f"{k} {sum(r[k]['success'] for r in results)}" for k in keys), flush=True)
    base = {(x["task_id"], x["init_index"]): x["success"]
            for x in map(json.loads, open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl"))}
    cells = collections.defaultdict(list)
    for r in results:
        cells[(r["target"], r["init"])].append(r)
    print(f"\n[summary] far -> near error (reach {100 * args.reach:.0f} cm) | follow | ideal alpha | follow minus base")
    for (f, n), key in zip(levels, keys):
        follow = np.mean([np.mean([x[key]["success"] for x in v]) for v in cells.values()])
        ideal = np.mean([base[c] or v[0][key]["success"] for c, v in cells.items()])
        print(f"  {100 * f:.1f} -> {100 * n:.1f} cm | {100 * follow:5.1f}% | {100 * ideal:5.1f}% | "
              f"{100 * (follow - np.mean(list(base.values()))):+5.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
