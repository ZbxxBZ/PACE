"""Diagnosis video for the 7.4 twin test (10-23): real scene (agentview, wrist) next to what pi0.5 was shown (the
twin's agentview, wrist) at every step of one cell; the twin view is the one of the latest query."""
import json
import socket
import sys

import cv2
import imageio
import numpy as np

import pro_common as C
import pro_oracle as P
import pro_twin as T
from pro_wire import recv_msg, send_msg

t, k = int(sys.argv[1]), int(sys.argv[2])
rows = [json.loads(l) for l in open(P.EXP + "/episodes.jsonl")]
arrays = np.load(P.EXP + "/actions.npz")
names = {i: n for n, i in P.task_index("libero_10_swap").items()}
r = next(r for r in sorted(rows, key=lambda r: r["init"]) if r["name"] == names[t] and r["success"])
key = "t%d_i%d" % (r["task"], r["init"])
rec = P.record_source(dict(key=key, task=r["task"], init=r["init"], seed=r["seed"], actions=arrays[key]))
evs = rec["events"]
plan = [(evs[i]["kind"] if i < len(evs) else None, a) for i, (_, _, a) in enumerate(P.phase_targets(rec))]
seed = 1000 + 100 * t + k
env, task, inits = C.make_env("libero_10_swap", t, C.SIZE)
twin = T.Twin(task.name, k, seed)
sock = socket.create_connection(("127.0.0.1", 18765))
obs = C.reset(env, inits[k], seed)
sim = P.Sim(env)
small = lambda x: cv2.resize(np.ascontiguousarray(x[::-1, ::-1]), (256, 256))
phase, run, calls, frames, tview, ok = 0, 0, 0, [], None, False
for step in range(520):
    if step % 10 == 0:
        kind, anchor = plan[phase]
        d = np.zeros(3)
        if anchor in twin.known:
            d[:2] = sim.positions()[anchor][:2] - twin.known[anchor][:2]
        twin.sync(env, d)
        tobs = twin.observation()
        tview = np.concatenate([small(tobs["agentview_image"]), small(tobs["robot0_eye_in_hand_image"])], axis=1)
        send_msg(sock, {"task": task.language, "seed": seed, "call": calls}, C.raw_fields(twin.env, tobs))
        _, a = recv_msg(sock)
        chunk, calls = a["actions"], calls + 1
        cv2.putText(tview, "pi0.5 sees: phase %d %s, d=%.0f cm" % (phase, anchor, 100 * np.hypot(d[0], d[1])),
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    act = np.asarray(chunk[step % 10], dtype=np.float64)
    obs, _, _, _ = env.step(act)
    real = np.concatenate([small(obs["agentview_image"]), small(obs["robot0_eye_in_hand_image"])], axis=1)
    cv2.putText(real, "real scene, step %d" % step, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    frames.append(np.concatenate([real, tview], axis=0))
    if env.check_success():
        ok = True
        break
    nxt = plan[phase][0]
    if nxt is not None and phase + 1 < len(plan):
        want = 1.0 if nxt == "grasp" else -1.0
        run = run + 1 if np.sign(act[6]) == want else 0
        if run >= 3:
            phase, run = phase + 1, 0
out = "/root/pace/runs/rectest/v10_pro/videos/twin_task%d_init%d.mp4" % (t, k)
imageio.mimwrite(out, frames, fps=30, codec="libx264", quality=8, macro_block_size=1)
print("VID", out, "success", ok, "steps", len(frames), "phase", phase, flush=True)
