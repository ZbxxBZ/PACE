"""
Does closed_loop.py's extra 224 render change what pi0.5 sees afterwards? (10-24) Two envs of the same cell and seed;
in env A call render224 at the first step (as the V10Env wrapper does before deciding), in env B never; then step both
with the same zero-ish actions and compare the observation images pi0.5 receives at every step. Also base seed 1 vs 2
agreement from the closed-loop log (pi0.5's own run-to-run variation on the same cells).
"""
import json

import numpy as np

import closed_loop as C
import collect_pairs as P
import v10_oracle_shift2 as O


def main():
    P.worker_init()
    G = O.G
    from lerobot.envs.configs import LiberoPlusEnv

    env_cfg = LiberoPlusEnv(task="libero_10")
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    tid = 934
    fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=tid, n_envs=1, camera_names=G["cams"],
                       episode_length=env_cfg.episode_length, init_states=env_cfg.init_states, gym_kwargs=gk,
                       control_mode=env_cfg.control_mode, camera_name_mapping=env_cfg.camera_name_mapping,
                       is_libero_plus=env_cfg.is_libero_plus)[0]
    ea, eb = fn.func(1, **fn.keywords), fn.func(1, **fn.keywords)
    oa, _ = ea.reset(seed=1000 + 7 * tid + 1)
    ob, _ = eb.reset(seed=1000 + 7 * tid + 1)
    diffs = []
    rng = np.random.default_rng(0)
    for t in range(30):
        if t % 10 == 0:
            C.render224(ea._env.env)  # the wrapper's extra render
        a = np.r_[rng.uniform(-0.3, 0.3, 6), -1.0]
        oa, *_ = ea.step(a)
        ob, *_ = eb.step(a)
        d = {k: float(np.abs(oa["pixels"][k].astype(int) - ob["pixels"][k].astype(int)).mean()) for k in oa["pixels"]}
        diffs.append(d)
    print("[render side effect] mean abs pixel diff of pi0.5's observations, steps 0..29:")
    for t in (0, 1, 5, 10, 11, 20, 29):
        print(f"  step {t}: {diffs[t]}")
    eps = [json.loads(l) for l in open("/root/pace/runs/v10b/closed/run1/episodes.jsonl")]
    b = {}
    for e in eps:
        if e["mode"] == "base":
            b.setdefault((e["task_id"], e["init"]), {})[e["seed"] - 7 * e["task_id"] - e["init"]] = e["success"]
    s1 = np.array([v[1000] for v in b.values()])
    s2 = np.array([v[5000] for v in b.values()])
    print(f"[base] seed 1 {s1.mean():.2f}, seed 2 {s2.mean():.2f}, disagree on {np.mean(s1 != s2):.2f} of cells; "
          f"seed 2 fails on {np.mean(~s2[s1]):.2f} of the cells seed 1 solved")


if __name__ == "__main__":
    main()
