"""
Does the learned d carry the previous phase's shift into the next phase? (10-25, run2 followed episodes)
For every followed V10-1b episode of runs/v10b/closed/run2 and every logged call t: the phase of t (the reference's
phases, v10_oracle_shift2.phase_targets), the true per-phase d (anchor displacement, shift_vector's phase values), the
network's d. Per phase index and outcome: median |d_hat - d_true(this phase)| and |d_hat - d_true(previous phase)|,
and the share of calls closer to the previous phase's d, over calls where the two phases' d differ by > 1.5 cm.
    python d_carryover.py
"""
import collections
import json
import multiprocessing as mp

import numpy as np

import collect_pairs as P
import v10_oracle_shift2 as O

RUN = "/root/pace/runs/v10b/closed/run2"
SRC = "/root/pace/runs/v10b/noise_sources"


def job(args):
    tid, k, seed, src, calls = args
    rec = O.record_source(src)
    phases = O.phase_targets(rec)
    tstart = O.target_start_positions(tid, k, seed)
    _, dinfo = O.shift_vector(rec, phases, tstart, len(src["actions"]), 15)
    dph = [np.array(x[3]) for x in dinfo]
    out = []
    for c in calls:
        t = c["t"]
        ph = next((i for i, (s, e, _) in enumerate(phases) if t < max(e, s + 1)), len(phases) - 1)
        if t < phases[ph][0] + 15 and ph > 0:  # inside the label's 15-step blend: skip
            continue
        dh = np.array(c["d"])
        out.append(dict(ph=ph, cur=float(np.linalg.norm(dh - dph[ph])),
                        prev=float(np.linalg.norm(dh - dph[ph - 1])) if ph > 0 else None,
                        gap=float(np.linalg.norm(dph[ph] - dph[ph - 1])) if ph > 0 else 0.0,
                        anchor=phases[ph][2]))
    return out


def main():
    P.worker_init()
    rows = [json.loads(l) for l in open(f"{SRC}/episodes.jsonl")]
    srcs = {int(r["task_id"]): dict(key=(r["base"], r["task_id"]), task_id=int(r["task_id"]), init_index=int(r["init"]),
                                    seed=int(r["seed"]), actions=np.load(f"{SRC}/{r['actions']}").astype(np.float64))
            for r in rows if r["success"]}
    eps = [json.loads(l) for l in open(f"{RUN}/episodes.jsonl")]
    jobs, meta = [], []
    for e in eps:
        if e["mode"] != "v10" or not e.get("follow"):
            continue
        w = json.load(open(f"{RUN}/wrapper/{e['key']}.json"))
        jobs.append((e["task_id"], e["init"], e["seed"], srcs[e["source"]], w["calls"]))
        meta.append(e)
    with mp.get_context("spawn").Pool(4, initializer=P.worker_init) as pool:
        res = pool.map(job, jobs)
    agg = collections.defaultdict(lambda: dict(cur=[], prev=[], closer=0, n=0))
    for e, out in zip(meta, res):
        for c in out:
            key = ("success" if e["success"] else "fail", c["ph"])
            a = agg[key]
            a["cur"].append(c["cur"])
            if c["prev"] is not None and c["gap"] > 0.015:
                a["prev"].append(c["prev"])
                a["n"] += 1
                a["closer"] += c["prev"] < c["cur"]
    print(f"[episodes] {len(meta)} followed ({sum(e['success'] for e in meta)} succeed)")
    print("outcome phase | median |d_hat - d_this| cm (calls) | over calls with |d_this - d_prev| > 1.5 cm: "
          "median |d_hat - d_prev| cm, share closer to d_prev")
    for (o, ph), a in sorted(agg.items()):
        s = (f"  {o:7s} {ph} | {100 * np.median(a['cur']):.2f} ({len(a['cur'])})")
        if a["n"]:
            s += f" | {100 * np.median(a['prev']):.2f}, {100 * a['closer'] / a['n']:.0f}% of {a['n']}"
        print(s)


if __name__ == "__main__":
    main()
