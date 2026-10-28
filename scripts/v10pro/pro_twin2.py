"""
Counterfactual observation, two corrections of pro_twin.py (section 7.5 of docs/PACE_V10_经验配准方案.md, registered
10-23 before running). pro_twin.py put every object the robot had not moved yet back where pi0.5 knows it and moved
only the hand by -d: the hand is right relative to the current anchor but wrong relative to every other object, so when
pi0.5 goes for a different object than the experience's order (e.g. tomato sauce before alphabet soup) the real hand
lands on whatever now stands there (the 7.4 video). Two ways to make the twin consistent:
  --mode shift_all : the documented design -- every movable object copied from the real scene shifted by -d (fixtures
                     stay where pi0.5 knows them), so hand-object geometry equals the real one for every object; only
                     the current anchor sits where pi0.5 remembers it.
  --mode intent    : the familiar layout for untouched objects (as pro_twin.py), but d follows pi0.5's intent: query
                     once, take the twin hand's planned xy motion over the chunk, pick the candidate (grasp phase:
                     untouched movables; place phase: every body but moved movables) best in line with it (cos > 0.7,
                     smallest distance to the ray); if that differs from the current target, d := its real - familiar
                     position, re-sync the twin and query again; small planned motion keeps the current target.
Everything else as pro_twin.py (100 cells libero_10_swap, base seeds, plans from the first experience, phase change on
pi0.5's gripper command for 3 steps, 520 steps). Upper bound with simulator truth.
    python pro_twin2.py --mode shift_all --workers 12 -> runs/rectest/v10_pro/twin2_<mode>_l10_swap.json
"""
import argparse
import json
import multiprocessing as mp
import random
import socket
import time

import numpy as np

import pro_common as C
import pro_oracle as P
import pro_twin as T
from pro_wire import recv_msg, send_msg

OUT = "/root/pace/runs/rectest/v10_pro/twin2_%s_l10_swap.json"


class Twin2(T.Twin):
    def sync(self, real_env, dvec, shift_all):
        if not shift_all:
            return super().sync(real_env, dvec)
        # the parent with every object counted as moved (its start positions put out of reach): each movable object
        # is copied from the real scene shifted by -d; fixtures, gripper, arm and IK unchanged
        if self.real0 is None:
            self.real0 = np.array(real_env.env.sim.data.qpos)
        start = self.real0
        self.real0 = np.full_like(start, 1e9)
        try:
            return super().sync(real_env, dvec)
        finally:
            self.real0 = start


def query(sock, env, twin, task, seed, calls):
    tobs = twin.observation()
    send_msg(sock, {"task": task.language, "seed": seed, "call": calls}, C.raw_fields(twin.env, tobs))
    _, a = recv_msg(sock)
    return np.asarray(a["actions"], dtype=np.float64)


def pick_target(chunk, hand_twin, cands, known, current):
    v = chunk[:, :2].sum(0)
    if np.linalg.norm(v) < 0.5 or not cands:
        return current
    u = v / np.linalg.norm(v)
    best, best_perp = current, 1e9
    for n in cands:
        w = known[n][:2] - hand_twin[:2]
        dist = np.linalg.norm(w)
        if dist < 1e-6:
            continue
        cos = float(w @ u) / dist
        perp = abs(w[0] * u[1] - w[1] * u[0])
        if cos > 0.7 and perp < best_perp:
            best, best_perp = n, perp
    return best


def run_cell(job):
    (t, k), plan, port, mode = job
    seed = 1000 + 100 * t + k
    env, task, inits = C.make_env("libero_10_swap", t, C.SIZE)
    twin = Twin2(task.name, k, seed)
    sock = socket.create_connection(("127.0.0.1", port))
    try:
        C.reset(env, inits[k], seed)
        sim = P.Sim(env)
        start = sim.positions()
        movable = set(env.env.objects_dict)
        phase, run, chunk, success, steps, calls, log = 0, 0, None, False, C.max_steps("libero_10_swap"), 0, []
        target = plan[0][1]
        for step in range(C.max_steps("libero_10_swap")):
            if step % 10 == 0:
                kind, anchor = plan[phase]
                pos = sim.positions()
                if mode == "shift_all" or target is None:
                    target = anchor
                d = np.zeros(3)
                if target in twin.known and target in pos:
                    d[:2] = pos[target][:2] - twin.known[target][:2]
                ik = twin.sync(env, d, mode == "shift_all")
                chunk = query(sock, env, twin, task, seed, calls)
                calls += 1
                switched = False
                if mode == "intent":
                    moved = {n for n in movable if np.linalg.norm(pos[n][:2] - start[n][:2]) > 0.01}
                    if kind == "grasp":
                        cands = [n for n in movable if n not in moved and n in twin.known]
                    else:
                        cands = [n for n in twin.known if n not in moved]
                    hand = np.array(env.env.sim.data.site_xpos[env.env.robots[0].eef_site_id]) - d
                    new = pick_target(chunk, hand, cands, twin.known, target)
                    if new != target and new in pos:
                        target, switched = new, True
                        d = np.zeros(3)
                        d[:2] = pos[target][:2] - twin.known[target][:2]
                        ik = twin.sync(env, d, False)
                        chunk = query(sock, env, twin, task, seed, calls)
                        calls += 1
                log.append([phase, target, round(float(np.hypot(d[0], d[1])), 3), round(ik, 4), switched])
            act = chunk[step % 10]
            env.step(act)
            if env.check_success():
                success, steps = True, step + 1
                break
            nxt = plan[phase][0]
            if nxt is not None and phase + 1 < len(plan):
                want = 1.0 if nxt == "grasp" else -1.0
                run = run + 1 if np.sign(act[6]) == want else 0
                if run >= 3:
                    phase, run = phase + 1, 0
                    target = None if mode == "intent" else target
        return dict(task=t, init=k, success=success, steps=steps, phase_reached=phase, n_phases=len(plan),
                    max_ik_err=max(x[3] for x in log), switches=sum(x[4] for x in log), log=log)
    finally:
        sock.close()
        twin.close()
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shift_all", "intent"], required=True)
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or OUT % args.mode
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    swap = P.task_index("libero_10_swap")
    names = {i: n for n, i in swap.items()}
    plans = {}
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        if r["success"] and r["name"] not in plans and swap[r["name"]] in args.tasks:
            key = "t%d_i%d" % (r["task"], r["init"])
            rec = P.record_source(dict(key=key, task=r["task"], init=r["init"], seed=r["seed"], actions=arrays[key]))
            assert rec["success"], key
            evs = rec["events"]
            plans[r["name"]] = [(evs[i]["kind"] if i < len(evs) else None, a) for i, (_, _, a) in
                                enumerate(P.phase_targets(rec))]
    jobs = [((t, k), plans[names[t]], args.port, args.mode) for t in args.tasks for k in args.inits]
    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            json.dump(results, open(out, "w"), indent=1)
            print(f"[progress] {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s): {args.mode} "
                  f"{sum(x['success'] for x in results)}/{len(results)}", flush=True)
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    sel = {(r["task"], r["init"]): r["success"] for r in results}
    tasks = sorted({c[0] for c in sel})
    rng = random.Random(0)
    diff = lambda ts: np.mean([sel[c] - base[c] for c in sel if c[0] in ts])
    bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
    summary = dict(mode=args.mode, base=round(100 * np.mean([base[c] for c in sel]), 1),
                   twin=round(100 * np.mean(list(sel.values())), 1),
                   twin_minus_base=[round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)],
                   per_task={f"{t} {names[t][:55]}": dict(base=round(100 * np.mean([base[c] for c in sel if c[0] == t])),
                                                          twin=round(100 * np.mean([sel[c] for c in sel if c[0] == t])))
                             for t in tasks})
    json.dump(summary, open(out.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
