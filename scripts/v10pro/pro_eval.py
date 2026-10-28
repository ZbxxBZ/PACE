"""
Closed-loop pi0.5 on LIBERO-PRO in its own code base (10-23): every (task, init state) of a suite once, the policy
served by pi05_server.py on the GPU, LIBERO-PRO's simulator with EGL on the GPU. Same reset and step budget as lerobot
(10 settle steps, libero_10: 520 steps), a new chunk of 10 actions every 10 steps, success from the env's own check.
Seed of an episode: 1000 + 100 * task + init (also seeds the policy's noise, so episodes are reproducible).
With --save_actions the executed env-unit actions of every success are kept: the experience for the true-d check.
    python pro_eval.py --suite libero_10 --inits 0 1 2 3 4 --save_actions --out runs/rectest/v10_pro/exp_l10
    python pro_eval.py --suite libero_10_swap --inits 0 1 2 3 4 5 6 7 8 9 --out runs/rectest/v10_pro/base_l10_swap
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import pro_common as C


def episode(job):
    suite, task_idx, init_idx, port = job
    seed = 1000 + 100 * task_idx + init_idx
    env, task, inits = C.make_env(suite, task_idx)
    client = C.PolicyClient(port)
    try:
        obs = C.reset(env, inits[init_idx], seed)
        acts, chunk, success, steps = [], None, False, C.max_steps(suite)
        for t in range(C.max_steps(suite)):
            if t % 10 == 0:
                chunk = client.act(env, obs, task.language, seed, t // 10)
            a = np.asarray(chunk[t % 10], dtype=np.float64)
            obs, _, _, _ = env.step(a)
            acts.append(a)
            if env.check_success():
                success, steps = True, t + 1
                break
        return dict(suite=suite, task=task_idx, name=task.name, language=task.language, init=init_idx, seed=seed,
                    success=success, steps=steps), np.stack(acts)
    finally:
        client.close()
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--save_actions", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [(args.suite, t, k, args.port) for t in args.tasks for k in args.inits]
    rows, saved, t0 = [], {}, time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool, open(f"{args.out}/episodes.jsonl", "w") as log:
        for row, acts in pool.imap_unordered(episode, jobs):
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            if args.save_actions and row["success"]:
                saved[f"t{row['task']}_i{row['init']}"] = acts
            print(f"[{len(rows)}/{len(jobs)}] task {row['task']} init {row['init']}: "
                  f"{'ok' if row['success'] else 'FAIL'} ({row['steps']} steps), "
                  f"{sum(r['success'] for r in rows)} ok so far ({time.time() - t0:.0f}s)", flush=True)
    if args.save_actions:
        np.savez_compressed(f"{args.out}/actions.npz", **saved)
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["language"]), []).append(r["success"])
    summary = dict(suite=args.suite, n=len(rows), success=sum(r["success"] for r in rows),
                   per_task={f"{t} {l}": f"{sum(v)}/{len(v)}" for (t, l), v in sorted(by.items())})
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
