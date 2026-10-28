"""
How accurate must d be? (10-23, PPT record table 3) The corrected true-d upper bound (v10_oracle_shift2.py: 20 camera
variants x init states 1..5, 3 sources each, the same seeds; sources re-collected with collect_pi05.py) rerun with an error added to every phase's d: a vector of
length --errs cm in a random direction, drawn once per (target, init, source, phase) from a fixed seed, before the
phase blending. Only the trackd condition; error 0 is rerun too so all levels share one code path.
    python v10_oracle_noise.py --workers 24 -> runs/rectest/v10_shift/oracle_noise.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import random
import time

import numpy as np

import v10_oracle_shift2 as O

OUT = "/root/pace/runs/rectest/v10_shift/oracle_noise.json"


def noisy_shift(rec, phases, target_start, n, blend, err, rng):
    """O.shift_vector with an error of length err (m) in a random direction added to each phase's d."""
    d_phase = []
    for _, _, b in phases:
        d = np.zeros(3)
        if b in target_start and b in rec["start"]:
            d[:2] = target_start[b][:2] - rec["start"][b][:2]
        th = rng.uniform(0, 2 * np.pi)
        d[:2] += err * np.array([np.cos(th), np.sin(th)])
        d_phase.append(d)
    dv = np.zeros((n, 3))
    prev = d_phase[0]
    for (s, e, _), d in zip(phases, d_phase):
        for t in range(s, min(max(e, s + 1), n)):
            w = min(1.0, (t - s) / max(blend, 1)) if s > 0 else 1.0
            dv[t] = (1 - w) * prev + w * d
        prev = d
    last = phases[-1][1]
    if last < n:
        dv[last:] = d_phase[-1]
    return dv


def job(args):
    (tid, k, seed), sources, kp, blend, errs = args
    tstart = O.target_start_positions(tid, k, seed)
    out = []
    for src in sources:
        rec = src["rec"]
        n = len(src["actions"])
        phases = O.phase_targets(rec)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        row = dict(target=tid, init=k, source=src["task_id"])
        for e in errs:
            rng = np.random.default_rng([tid, k, src["task_id"], int(round(e * 1000))])
            dvec = noisy_shift(rec, phases, tstart, n, blend, e, rng)
            ok, steps = O.run_episode(tid, k, seed, src["actions"], "track", path, dvec, kp)
            row[f"err{int(round(e * 100))}"] = dict(success=ok, steps=steps)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errs", type=float, nargs="+", default=[0.0, 0.01, 0.02, 0.03])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sources", default="/root/pace/runs/v10b/noise_sources")
    args = ap.parse_args()
    # sources: pi0.5 successes on light / noise / background variants from init 0 (collect_pi05.py --set
    # noise_sources; the 2x3090's V10-0 run was unreachable 10-23), targets: the 20 camera variants x init 1..5
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
                jobs.append(((tid, k, 1000 + 7 * tid + k), pool_src, 1.0, 15, args.errs))
        print(f"[jobs] {len(jobs)} x 3 sources x errors {args.errs} ({time.time() - t0:.0f}s)", flush=True)
        results = []
        for out in pool.imap_unordered(job, jobs):
            results += out
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"[progress] {len(results)} pairs ({time.time() - t0:.0f}s): " + ", ".join(
                f"{int(round(e * 100))}cm {sum(r[f'err{int(round(e * 100))}']['success'] for r in results)}"
                for e in args.errs), flush=True)
    base = {(x["task_id"], x["init_index"]): x["success"]
            for x in map(json.loads, open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl"))}
    cells = collections.defaultdict(list)
    for r in results:
        cells[(r["target"], r["init"])].append(r)
    print("\n[summary] error | follow (mean over sources) | ideal alpha (base or the first source) | minus base")
    for e in args.errs:
        key = f"err{int(round(e * 100))}"
        follow = np.mean([np.mean([x[key]["success"] for x in v]) for v in cells.values()])
        ideal = np.mean([base[c] or v[0][key]["success"] for c, v in cells.items()])
        print(f"  {100 * e:3.0f} cm | {100 * follow:5.1f}% | {100 * ideal:5.1f}% | {100 * (follow - np.mean(list(base.values()))):+5.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
