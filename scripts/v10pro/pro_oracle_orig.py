"""
In-distribution check (10-23, user: "两个一起跑"): does following experience hurt on the ORIGINAL libero_10, where
pi0.5 already succeeds 95/100 (exp_l10)? Same conditions as pro_oracle.py (replay / track0 / trackd with the true d,
corrected anchors), but targets are the original libero_10 tasks x init states 0..9 (the base run's seeds), and the
experiences of a target come from the same task's OTHER init states (never its own episode: no exact copy).
Reported against pi0.5 on the same cells (exp_l10): always-follow (trackd, mean over experiences), damage on the base's
successes, and the ideal alpha (per cell: pi0.5's result, or trackd with the first experience when pi0.5 failed).
    python pro_oracle_orig.py --workers 12 -> runs/rectest/v10_pro/oracle_l10_orig.json
"""
import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import pro_common as C
import pro_oracle as P

OUT = "/root/pace/runs/rectest/v10_pro/oracle_l10_orig.json"


def patch():
    make = C.make_env
    C.make_env = lambda suite, idx, size=C.SIZE: make("libero_10" if suite == "libero_10_swap" else suite, idx, size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--per_target", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    patch()
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    base = {(r["task"], r["init"]): r["success"] for r in rows}
    srcs = [dict(key=f"t{r['task']}_i{r['init']}", task=r["task"], name=r["name"], init=r["init"], seed=r["seed"],
                 actions=arrays[f"t{r['task']}_i{r['init']}"])
            for r in sorted(rows, key=lambda r: (r["task"], r["init"])) if r["success"] and f"t{r['task']}_i{r['init']}" in arrays.files]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=patch) as pool:
        recs = dict(pool.map(P.rec_job, srcs))
        for s in srcs:
            s["rec"] = recs[s["key"]]
        good = [s for s in srcs if s["rec"]["success"]]
        print(f"[experience] {len(srcs)} replayed, {len(good)} succeed again ({time.time() - t0:.0f}s)", flush=True)
        jobs = []
        for t in range(10):
            for k in args.inits:
                pool_src = [s for s in good if s["task"] == t and s["init"] != k][:args.per_target]
                jobs.append(((t, k), pool_src, 1.0, 15))
        results = []
        for out in pool.imap_unordered(P.job, jobs):
            results += out
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"[progress] {len(results)} pairs ({time.time() - t0:.0f}s): " + ", ".join(
                f"{m} {sum(r[m]['success'] for r in results)}" for m in ("replay", "track0", "trackd")), flush=True)
    cells = {}
    for r in results:
        cells.setdefault((r["target"], r["init"]), []).append(r)
    follow = np.mean([np.mean([x["trackd"]["success"] for x in v]) for v in cells.values()])
    succ = [c for c in cells if base[c]]
    damage = 1 - np.mean([np.mean([x["trackd"]["success"] for x in cells[c]]) for c in succ])
    ideal = np.mean([base[c] or cells[c][0]["trackd"]["success"] for c in cells])
    summary = dict(cells=len(cells), base=round(100 * np.mean([base[c] for c in cells]), 1),
                   always_follow=round(100 * follow, 1), damage_on_base_successes=round(100 * damage, 1),
                   ideal_alpha=round(100 * ideal, 1),
                   replay=round(100 * np.mean([x["replay"]["success"] for v in cells.values() for x in v]), 1),
                   per_task={t: dict(base=round(100 * np.mean([base[c] for c in cells if c[0] == t])),
                                     follow=round(100 * np.mean([x["trackd"]["success"] for c in cells if c[0] == t
                                                                 for x in cells[c]])))
                             for t in range(10)})
    json.dump(summary, open(OUT.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
