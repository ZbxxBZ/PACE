"""
Section 7.3 of docs/PACE_V10_经验配准方案.md (registered 10-23 17:40): shift pi0.5's proprioceptive frame instead of
moving the hand. Upper bound with simulator truth on libero_10_swap (10 tasks x init states 0..9, the base run's seeds).
"The layout pi0.5 knows" = the original libero_10 scene with the same init state index (positions after the settle
steps). Phase order and anchors come from the task's first experience (gripper events, the corrected anchor rule of
pro_oracle.py). In phase i, d_i = live position of anchor_i in the swapped scene - its position in the original
scene (xy); every time pi0.5 is queried, the eef position it receives is eef - d_i, everything else (images, eef
orientation, gripper, joints) unchanged, and its actions are executed as they are. The phase advances when pi0.5
commands the next event's gripper change for 3 steps in a row; the new d applies from the next query. 520 steps.
    python pro_shift.py --workers 10 -> runs/rectest/v10_pro/shift_l10_swap.json
"""
import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

import pro_common as C
import pro_oracle as P
from pro_wire import recv_msg, send_msg

OUT = "/root/pace/runs/rectest/v10_pro/shift_l10_swap.json"


def known_positions(task_name, init_idx, seed):
    """Body positions after the settle steps in the original libero_10 scene with the same init state index."""
    idx = P.task_index("libero_10")[task_name]
    env, _, inits = C.make_env("libero_10", idx, P.SIZE)
    try:
        C.reset(env, inits[init_idx], seed)
        return P.Sim(env).positions()
    finally:
        env.close()


def run_cell(job):
    (t, k), plan, port = job
    seed = 1000 + 100 * t + k
    env, task, inits = C.make_env("libero_10_swap", t, C.SIZE)
    known = known_positions(task.name, k, seed)
    import socket
    sock = socket.create_connection(("127.0.0.1", port))
    try:
        obs = C.reset(env, inits[k], seed)
        sim = P.Sim(env)
        phase, run, chunk, success, steps, calls, used = 0, 0, None, False, C.max_steps("libero_10_swap"), 0, []
        for step in range(C.max_steps("libero_10_swap")):
            if step % 10 == 0:
                kind, anchor = plan[phase]
                d = np.zeros(3)
                if anchor in known:
                    d[:2] = sim.positions()[anchor][:2] - known[anchor][:2]
                fields = C.raw_fields(env, obs)
                fields["eef_pos"] = np.asarray(fields["eef_pos"]) - d
                send_msg(sock, {"task": task.language, "seed": seed, "call": calls}, fields)
                _, a = recv_msg(sock)
                chunk = a["actions"]
                calls += 1
                used.append([phase, round(float(d[0]), 4), round(float(d[1]), 4)])
            act = np.asarray(chunk[step % 10], dtype=np.float64)
            obs, _, _, _ = env.step(act)
            if env.check_success():
                success, steps = True, step + 1
                break
            nxt = plan[phase][0]
            if nxt is not None and phase + 1 < len(plan):
                want = 1.0 if nxt == "grasp" else -1.0
                run = run + 1 if np.sign(act[6]) == want else 0
                if run >= 3:
                    phase, run = phase + 1, 0
        return dict(task=t, init=k, success=success, steps=steps, phase_reached=phase, n_phases=len(plan),
                    d_per_query=used)
    finally:
        sock.close()
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    swap = P.task_index("libero_10_swap")
    plans = {}
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        if r["success"] and r["name"] not in plans:
            key = "t%d_i%d" % (r["task"], r["init"])
            rec = P.record_source(dict(key=key, task=r["task"], init=r["init"], seed=r["seed"], actions=arrays[key]))
            assert rec["success"], key
            evs = rec["events"]
            plans[r["name"]] = [(evs[i]["kind"] if i < len(evs) else None, a) for i, (_, _, a) in
                                enumerate(P.phase_targets(rec))]
    names = {i: n for n, i in swap.items()}
    print("[plans]", {i: plans[names[i]] for i in args.tasks}, flush=True)
    jobs = [((t, k), plans[names[t]], args.port) for t in args.tasks for k in args.inits]
    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)
            print(f"[progress] {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s): shift "
                  f"{sum(x['success'] for x in results)}/{len(results)}", flush=True)
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    sel = {(r["task"], r["init"]): r["success"] for r in results}
    tasks = sorted({c[0] for c in sel})
    rng = random.Random(0)
    diff = lambda ts: np.mean([sel[c] - base[c] for c in sel if c[0] in ts])
    bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
    summary = dict(base=round(100 * np.mean([base[c] for c in sel]), 1), shift=round(100 * np.mean(list(sel.values())), 1),
                   shift_minus_base=[round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)],
                   per_task={f"{t} {names[t][:55]}": dict(base=round(100 * np.mean([base[c] for c in sel if c[0] == t])),
                                                          shift=round(100 * np.mean([sel[c] for c in sel if c[0] == t])))
                             for t in tasks})
    json.dump(summary, open(args.out.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
