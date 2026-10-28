"""
V10-1b training pairs, simulation only (10-24; docs/PACE_V1-V8.md "V10-1b 结构与数据采集方案", steps 2-4). No policy,
no images: every episode keeps what an exact replay needs (the executed actions) plus the per-step labels, and frames are
rendered later (with augmentation) by replaying them.
References: the pi0.5 successes on the original libero_10 tasks (collect_pi05.py --set orig -> runs/v10b/refs), each
first replayed in its own scene (must succeed again) to record its hand path, gripper events and body positions
(v10_oracle_shift2.record_source); the replays are cached in <out>/refs_rec.pkl.
Current scenes: the same 10 original tasks x init states 6..49 (45..49 = validation: their references come only from
45..49, training scenes' only from 6..44). Per current scene:
  expert    : --per_scene references of the same task (other init states), followed with the true d (BDDL anchors,
              v10_oracle_shift2.phase_targets / shift_vector);
  perturbed : the same pairs, followed with true d + an error of U(--err_lo, --err_hi) m in a random direction per phase
              (blended across phases like d), so the network sees the hand off the line; the label stays the true d;
  negative  : one reference of ANOTHER task (the same scene's other task when there is one, LIVING_ROOM_SCENE2; else
              a random other task), followed with d from the anchors both scenes share (0 otherwise); alpha label 0.
Per episode <out>/<kind>/<key>.npz: actions (executed, env units), d (true, per step), err (added, per step), path
(the reference's hand path, per step), phases; one row per episode in <out>/<kind>.jsonl.
    python collect_pairs.py --workers 12 --out runs/v10b/pairs
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import pickle
import re
import time

import numpy as np

import collect_pi05 as C
import v10_oracle_shift2 as O

VAL = set(range(45, 50))


def worker_init():
    O.worker_init()
    O.G["suite"] = C.ComboSuite(O.G["suite"], C.original_tasks(O.G["suite"]))


def blended(phases, vecs, n, blend=15):
    """Per-step vector from one vector per phase, blended into the next over `blend` steps (as O.shift_vector)."""
    dv = np.zeros((n, 3))
    prev = vecs[0]
    for (s, e, _), d in zip(phases, vecs):
        for t in range(s, min(max(e, s + 1), n)):
            w = min(1.0, (t - s) / max(blend, 1)) if s > 0 else 1.0
            dv[t] = (1 - w) * prev + w * d
        prev = d
    if phases[-1][1] < n:
        dv[phases[-1][1]:] = vecs[-1]
    return dv


def follow(tid, k, seed, ref_actions, path, dvec, hold=40):
    """O.run_episode's track mode, also returning the executed actions."""
    env = O.make_env(tid, k)
    try:
        env.reset(seed=int(seed))
        sim = O.Sim(env)
        n, out = len(ref_actions), []
        for t in range(n + hold):
            a = ref_actions[t].copy() if t < n else np.r_[np.zeros(6), ref_actions[-1][6]]
            tgt = path[min(t, n - 1)] + dvec[min(t, n - 1)]
            a[:3] = np.clip(a[:3] + (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            out.append(a)
            _, _, _, _, info = env.step(a)
            if info.get("is_success"):
                return True, np.stack(out)
        return False, np.stack(out)
    finally:
        env.close()


def rec_job(r):
    src = dict(key=(r["task_id"], r["init"]), task_id=r["task_id"], init_index=r["init"], seed=r["seed"],
               actions=r["acts"])
    return src["key"], O.record_source(src)


def job(args):
    kind, cur, ref, refrec, err_range, out_dir = args
    tid, k, seed = cur["task_id"], cur["init"], cur["seed"]
    key = f"t{tid}_i{k}__t{ref['task_id']}_i{ref['init']}"
    if os.path.exists(f"{out_dir}/{kind}/{key}.npz"):
        return None
    acts = ref["acts"]
    n = len(acts)
    phases = O.phase_targets(refrec)
    tstart = O.target_start_positions(tid, k, seed)
    dvec, dinfo = O.shift_vector(refrec, phases, tstart, n, 15)
    path = np.concatenate([refrec["path"], np.repeat(refrec["path"][-1:], max(0, n - len(refrec["path"])), 0)])
    err = np.zeros((n, 3))
    if kind == "carry":  # 10-25 (V10-3): the hand enters each phase still shifted by the PREVIOUS phase's d (what
        # the learned d did in closed loop, d_carryover.py); phase 0 is pushed off like "perturbed"; label = true d
        rng = np.random.default_rng([tid, k, ref["task_id"], ref["init"], 7])
        th, m = rng.uniform(0, 2 * np.pi), rng.uniform(*err_range)
        vecs = [np.array([m * np.cos(th), m * np.sin(th), 0.0])]
        for i in range(1, len(phases)):
            vecs.append(np.r_[np.array(dinfo[i - 1][3]) - np.array(dinfo[i][3]), 0.0])
        err = blended(phases, vecs, n)
    if kind == "perturbed":
        rng = np.random.default_rng([tid, k, ref["task_id"], ref["init"]])
        vecs = []
        for _ in phases:
            th, m = rng.uniform(0, 2 * np.pi), rng.uniform(*err_range)
            vecs.append(np.array([m * np.cos(th), m * np.sin(th), 0.0]))
        err = blended(phases, vecs, n)
    ok, executed = follow(tid, k, seed, acts, path, dvec + err)
    np.savez_compressed(f"{out_dir}/{kind}/{key}.npz", actions=executed.astype(np.float32), d=dvec.astype(np.float32),
                        err=err.astype(np.float32), path=path.astype(np.float32),
                        phases=np.array([(s, e) for s, e, _ in phases]), anchors=np.array([b for _, _, b in phases]))
    return dict(kind=kind, key=key, task_id=tid, init=k, seed=seed, ref_task=ref["task_id"], ref_init=ref["init"],
                ref_seed=ref["seed"], split="val" if k in VAL else "train", success=ok, steps=len(executed),
                phases=dinfo, err_max=float(np.abs(err[:, :2]).max()) if kind in ("perturbed", "carry") else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="/root/pace/runs/v10b/refs")
    ap.add_argument("--out", default="/root/pace/runs/v10b/pairs")
    ap.add_argument("--per_scene", type=int, default=3)
    ap.add_argument("--err_lo", type=float, default=0.01)
    ap.add_argument("--err_hi", type=float, default=0.03)
    ap.add_argument("--kinds", nargs="+", default=["expert", "perturbed", "negative"])
    ap.add_argument("--limit", type=int, default=0, help="current scenes per task (0 = all), for a pilot")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3000)
    args = ap.parse_args()
    for kd in args.kinds:
        os.makedirs(f"{args.out}/{kd}", exist_ok=True)
    rows = [json.loads(l) for l in open(f"{args.refs}/episodes.jsonl")]
    for r in rows:
        r["acts"] = np.load(f"{args.refs}/{r['actions']}").astype(np.float64)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=worker_init) as pool:
        cache = f"{args.out}/refs_rec.pkl"
        recs = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
        todo = [r for r in rows if r["success"] and (r["task_id"], r["init"]) not in recs]
        if todo:
            recs.update(pool.map(rec_job, todo))
            pickle.dump(recs, open(cache, "wb"))
        refs = collections.defaultdict(list)
        for r in rows:
            if r["success"] and recs[(r["task_id"], r["init"])]["success"]:
                refs[r["task_id"]].append(r)
        print(f"[refs] {sum(map(len, refs.values()))} usable references "
              f"(of {sum(r['success'] for r in rows)} pi0.5 successes; replay must succeed again) "
              f"({time.time() - t0:.0f}s)", flush=True)
        scene = {tid: re.match(r".*?SCENE\d+", rs[0]["name"]).group(0) for tid, rs in refs.items()}
        jobs = []
        for tid in sorted(refs):
            inits = sorted({r["init"] for r in rows if r["task_id"] == tid})
            if args.limit:
                inits = inits[:args.limit]
            for j, k in enumerate(inits):
                cur = dict(task_id=tid, init=k, seed=args.seed + 100 * (tid - C.OFFSET) + k)
                rng = np.random.default_rng([tid, k])
                same = [r for r in refs[tid] if r["init"] != k and ((r["init"] in VAL) == (k in VAL))]
                pick = [same[i] for i in rng.choice(len(same), min(args.per_scene, len(same)), replace=False)]
                for kd in ("expert", "perturbed", "carry"):
                    if kd in args.kinds:
                        jobs += [(kd, cur, r, recs[(r["task_id"], r["init"])], (args.err_lo, args.err_hi), args.out)
                                 for r in pick]
                if "negative" in args.kinds:
                    others = [t for t in refs if t != tid and scene[t] == scene[tid]] or [t for t in refs if t != tid]
                    ot = others[int(rng.integers(len(others)))]
                    cand = [r for r in refs[ot] if (r["init"] in VAL) == (k in VAL)]
                    r = cand[int(rng.integers(len(cand)))]
                    jobs.append(("negative", cur, r, recs[(r["task_id"], r["init"])], (0, 0), args.out))
        print(f"[jobs] {len(jobs)}: " + ", ".join(f"{kd} {sum(j[0] == kd for j in jobs)}" for kd in args.kinds),
              flush=True)
        logs = {kd: open(f"{args.out}/{kd}.jsonl", "a") for kd in args.kinds}
        cnt = collections.Counter()
        for i, row in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            if row is not None:
                logs[row["kind"]].write(json.dumps(row) + "\n")
                logs[row["kind"]].flush()
                cnt[row["kind"], row["success"]] += 1
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[progress] {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s): " + ", ".join(
                    f"{kd} {cnt[kd, True]}/{cnt[kd, True] + cnt[kd, False]}" for kd in args.kinds), flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
