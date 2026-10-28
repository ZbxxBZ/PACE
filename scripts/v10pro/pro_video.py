"""
Videos of gate 1 on LIBERO-PRO's position swap (10-23): one libero_10_swap scene (task, init state, seed as in the
base run and the true-d check) run three ways, stacked:
  row 1: pi0.5 itself (served by pi05_server.py; same seed, so the same episode as the base run);
  row 2: replay   -- the experience's actions, open loop;
  row 3: trackd   -- follow the experience's hand path shifted per phase by the true displacement (pro_oracle.py).
Each row: agentview | wrist camera (256 px, flipped like lerobot's render), outcome stamped at the end. EGL on the GPU.
    python pro_video.py --cells 2:0 3:0 4:0 7:0 8:0 9:0 1:0 -> runs/rectest/v10_pro/videos/*.mp4
"""
import argparse
import json
import os

import cv2
import imageio
import numpy as np

import pro_common as C
import pro_oracle as P

OUT = "/root/pace/runs/rectest/v10_pro/videos"
SIZE = 256


def frame(obs):
    cams = [np.ascontiguousarray(obs[k][::-1, ::-1]) for k in ("agentview_image", "robot0_eye_in_hand_image")]
    return np.concatenate(cams, axis=1)


def label(img, text, color):
    bar = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


def run(task_idx, init_idx, seed, mode, actions=None, path=None, dvec=None, client=None, hold=40, tail=20):
    env, task, inits = C.make_env("libero_10_swap", task_idx, SIZE)
    try:
        obs = C.reset(env, inits[init_idx], seed)
        sim = P.Sim(env)
        frames, ok = [frame(obs)], False
        n = C.max_steps("libero_10_swap") if mode == "policy" else len(actions) + hold
        chunk = None
        for t in range(n):
            if mode == "policy":
                if t % 10 == 0:
                    chunk = client.act(env, obs, task.language, seed, t // 10)
                a = np.asarray(chunk[t % 10], dtype=np.float64)
            else:
                m = len(actions)
                if t < m:
                    a = np.array(actions[t], dtype=np.float64)
                else:
                    a = np.zeros(7)
                    a[6] = actions[-1][6]
                if mode == "track":
                    tgt = path[min(t, m - 1)] + dvec[min(t, m - 1)]
                    a[:3] = np.clip(a[:3] + (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            obs, _, _, _ = env.step(a)
            frames.append(frame(obs))
            if env.check_success():
                ok = True
                break
        last = np.zeros(7)
        last[6] = a[6]
        for _ in range(tail):
            obs, _, _, _ = env.step(last)
            frames.append(frame(obs))
        return frames, ok, task
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["2:0", "3:0", "4:0", "7:0", "8:0", "9:0", "1:0"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    oracle = json.load(open(P.OUT))
    exp_rows = {f"t{r['task']}_i{r['init']}": r for r in map(json.loads, open(f"{P.EXP}/episodes.jsonl"))}
    arrays = np.load(f"{P.EXP}/actions.npz")
    client = C.PolicyClient(args.port)
    summary = []
    for cell in args.cells:
        t, k = map(int, cell.split(":"))
        seed = 1000 + 100 * t + k
        orow = next(r for r in oracle if r["target"] == t and r["init"] == k)
        er = exp_rows[orow["source"]]
        src = dict(key=orow["source"], task=er["task"], init=er["init"], seed=er["seed"], actions=arrays[orow["source"]])
        rec = P.record_source(src)
        n = len(src["actions"])
        phases = P.phase_targets(rec)
        dvec, dinfo = P.shift_vector(rec, phases, P.target_start(t, k, seed), n, 15)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        f_pol, ok_pol, task = run(t, k, seed, "policy", client=client)
        f_rep, ok_rep, _ = run(t, k, seed, "replay", src["actions"])
        f_trk, ok_trk, _ = run(t, k, seed, "track", src["actions"], path, dvec)
        T = max(len(f_pol), len(f_rep), len(f_trk))
        pad = lambda f: f + [f[-1]] * (T - len(f))
        f_pol, f_rep, f_trk = pad(f_pol), pad(f_rep), pad(f_trk)
        dmax = max(np.hypot(*p[3]) for p in dinfo) * 100
        name = task.name.split("SCENE")[-1][2:42]
        out = f"{OUT}/task{t}_init{k}_{name}.mp4"
        rows = [("pi0.5 itself", f_pol, ok_pol), ("Replay the experience", f_rep, ok_rep),
                (f"Follow + true d (max {dmax:.0f} cm)", f_trk, ok_trk)]
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as w:
            for i in range(T):
                end = i >= T - 25
                parts = []
                for title, fr, ok in rows:
                    col = ((20, 150, 40) if ok else (200, 30, 30)) if end else (40, 40, 40)
                    parts.append(label(fr[i], title + ((" -> SUCCESS" if ok else " -> FAIL") if end else ""), col))
                w.append_data(np.concatenate(parts, axis=0))
        summary.append(dict(task=t, init=k, name=task.name, source=orow["source"], pi05=ok_pol, replay=ok_rep,
                            trackd=ok_trk, phases=dinfo, video=out))
        print(f"[video] task {t} init {k}: pi0.5 {ok_pol}, replay {ok_rep}, trackd {ok_trk}, max d {dmax:.0f} cm -> {out}",
              flush=True)
    client.close()
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
