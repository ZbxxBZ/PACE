"""
Videos for the LIBERO-plus true-d check (10-23): the same target scene (a camera-viewpoint variant from init state k,
objects 2-4 cm away from where the experience succeeded) run twice with the same experience (a V10-0 success of a
light / noise / background variant from init state 0):
  top    : replay   -- the experience's actions, open loop (before correction);
  bottom : trackd   -- follow the experience's hand path shifted per phase by the true displacement of the phase's goal
                       anchor (v10_oracle_shift2's corrected rule; after correction).
Each row: agentview | wrist camera, 256 px, with the outcome stamped at the end. EGL on the GPU.
    python v10_video_pairs.py -> runs/rectest/v10_shift/videos/*.mp4
"""
import argparse
import json
import os
import sys

import cv2
import imageio
import numpy as np

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
import v10_oracle_shift2 as O  # noqa: E402

OUT = "/root/pace/runs/rectest/v10_shift/videos"
SRC = "/root/pace/runs/rectest/v10_pro/sources_l10"
PAIRS = [  # (target variant, init state, source variant, short name)
    (840, 1, 2441, "white_mug_two_plates"),
    (1098, 1, 2349, "mug_in_microwave"),
    (895, 3, 2502, "book_in_caddy"),
    (946, 5, 2484, "mug_plate_pudding"),
    (684, 5, 2409, "soup_sauce_basket"),
]
SIZE = 256


def make_env(tid, init_index):
    c = O.G["env_cfg"]
    fn = O.G["make_fns"](suite=O.G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=O.G["cams"],
                         episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=O.G["gk"],
                         control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    kw = dict(fn.keywords)
    kw.update(observation_width=SIZE, observation_height=SIZE)
    return fn.func(int(init_index), **kw)


def frame(obs):
    px = obs["pixels"]
    cams = [np.ascontiguousarray(px[k][::-1, ::-1]) for k in sorted(px)]  # image (agentview), image2 (wrist)
    return np.concatenate(cams, axis=1)


def label(img, text, color):
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def episode(tid, k, seed, actions, mode, path=None, dvec=None, kp=1.0, hold=40, tail=20):
    env = make_env(tid, k)
    try:
        obs, _ = env.reset(seed=int(seed))
        sim = O.Sim(env)
        frames, n, ok = [frame(obs)], len(actions), False
        for t in range(n + hold):
            if t < n:
                a = actions[t].copy()
            else:
                a = np.zeros(7)
                a[6] = actions[-1][6]
            if mode == "track":
                tgt = path[min(t, n - 1)] + dvec[min(t, n - 1)]
                a[:3] = np.clip(a[:3] + kp * (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            obs, _, _, _, info = env.step(a)
            frames.append(frame(obs))
            if info.get("is_success"):
                ok = True
                break
        last = a.copy()
        last[:6] = 0
        for _ in range(tail):  # a short still at the end
            obs, _, _, _, _ = env.step(last)
            frames.append(frame(obs))
        return frames, ok
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    O.worker_init()
    meta = {m["task_id"]: m for m in json.load(open(SRC + ".json"))}
    arrays = np.load(SRC + ".npz")
    summary = []
    for tid, k, src_tid, name in PAIRS:
        m = meta[src_tid]
        src = dict(task_id=src_tid, init_index=m["init_index"], seed=m["seed"], actions=arrays[m["key"]])
        rec = O.record_source(src)
        assert rec["success"], f"source {src_tid} does not succeed again"
        n = len(src["actions"])
        seed = 1000 + 7 * tid + k
        phases = O.phase_targets(rec)
        dvec, dinfo = O.shift_vector(rec, phases, O.target_start_positions(tid, k, seed), n, 15)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        f_rep, ok_rep = episode(tid, k, seed, src["actions"], "replay")
        f_trk, ok_trk = episode(tid, k, seed, src["actions"], "track", path, dvec)
        T = max(len(f_rep), len(f_trk))
        pad = lambda f: f + [f[-1]] * (T - len(f))
        f_rep, f_trk = pad(f_rep), pad(f_trk)
        red, green = (200, 30, 30), (20, 150, 40)
        shift = max(np.hypot(*p[3]) for p in dinfo) * 100
        out = f"{OUT}/{name}_target{tid}_init{k}.mp4"
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as w:
            for t in range(T):
                top = label(f_rep[t], "Before: replay the experience" +
                            (f"  ->  {'SUCCESS' if ok_rep else 'FAIL'}" if t >= T - 25 else ""),
                            green if (ok_rep and t >= T - 25) else red if t >= T - 25 else (40, 40, 40))
                bot = label(f_trk[t], f"After: shift by true d (max {shift:.1f} cm)" +
                            (f"  ->  {'SUCCESS' if ok_trk else 'FAIL'}" if t >= T - 25 else ""),
                            green if (ok_trk and t >= T - 25) else red if t >= T - 25 else (40, 40, 40))
                w.append_data(np.concatenate([top, bot], axis=0))
        summary.append(dict(name=name, target=tid, init=k, source=src_tid, replay=ok_rep, trackd=ok_trk,
                            max_shift_cm=round(shift, 1), phases=dinfo, video=out))
        print(f"[video] {name}: replay {'ok' if ok_rep else 'FAIL'}, trackd {'ok' if ok_trk else 'FAIL'}, "
              f"max shift {shift:.1f} cm, {T} frames -> {out}", flush=True)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
