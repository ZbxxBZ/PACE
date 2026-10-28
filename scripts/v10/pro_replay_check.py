"""
Step 1 of the LIBERO-PRO true-d check (10-23, docs/PACE_V10_经验配准方案.md): do the exported experiences (env-unit
action sequences of V10-0 successes from the original libero_10 layout, init state 0; sources_l10.npz) succeed again
when replayed open loop in LIBERO-PRO's own code base (Python 3.8, robosuite 1.4.0) on its original libero_10 tasks?
The reset follows lerobot's LiberoEnv: env built once and reset, then per episode seed -> reset -> set_init_state(init 0)
-> 10 no-op settle steps (gripper open), then the actions; success from the env's own check. EGL on the GPU.
    source /root/pace/env_pro_4090.sh; python pro_replay_check.py --workers 12
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

SRC = "/root/pace/runs/rectest/v10_pro/sources_l10"
OUT = "/root/pace/runs/rectest/v10_pro/replay_check.json"


def norm(s):
    return " ".join(s.lower().replace("_", " ").split())


def task_table():
    """normalized instruction -> (task index, bddl path, init states) of LIBERO-PRO's original libero_10."""
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    out = {}
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        out[norm(t.language)] = (i, os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                                 suite.get_task_init_states(i))
    return out


def replay(job):
    from libero.libero.envs import OffScreenRenderEnv

    key, lang, seed, init_index, acts = job
    _, bddl, inits = task_table()[lang]
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128, control_freq=20)
    try:
        env.reset()
        env.seed(seed)
        env.reset()
        env.set_init_state(inits[init_index])
        for _ in range(10):
            env.step([0, 0, 0, 0, 0, 0, -1])
        for t, a in enumerate(acts):
            env.step(a)
            if env.check_success():
                return key, True, t + 1
        return key, False, len(acts)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    meta = json.load(open(SRC + ".json"))
    arrays = np.load(SRC + ".npz")
    table = task_table()
    missing = sorted({norm(m["base"]) for m in meta} - set(table))
    assert not missing, f"no LIBERO-PRO libero_10 task for {missing}"
    jobs = [(m["key"], norm(m["base"]), m["seed"], m["init_index"], arrays[m["key"]]) for m in meta]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = {k: (ok, n) for k, ok, n in pool.imap_unordered(replay, jobs)}
    rows = [dict(m, success=res[m["key"]][0], replay_steps=res[m["key"]][1]) for m in meta]
    json.dump(rows, open(OUT, "w"), indent=1)
    ok = sum(r["success"] for r in rows)
    print(f"[replay] {ok}/{len(rows)} succeed again ({time.time() - t0:.0f}s)")
    by = {}
    for r in rows:
        by.setdefault(r["base"][:55], []).append(r["success"])
    for b, v in sorted(by.items()):
        print(f"  {sum(v)}/{len(v)}  {b}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
