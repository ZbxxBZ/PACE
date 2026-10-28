"""Stage-test diagnosis (10-23): per phase, where the hand ends after staging vs. the target, and the anchor's height
change during the contact (an object lifted = a real grasp). Task 1 init 0, "exp" contact."""
import json

import numpy as np

import pro_oracle as P
import pro_stage as S

oracle = json.load(open(P.OUT))
print("DBG oracle trackd at task 1 init 0:", [(r["source"], r["trackd"]["success"]) for r in oracle
                                              if r["target"] == 1 and r["init"] == 0], flush=True)
rows = [json.loads(l) for l in open(P.EXP + "/episodes.jsonl")]
arr = np.load(P.EXP + "/actions.npz")
r = next(r for r in sorted(rows, key=lambda r: r["init"]) if r["task"] == 1 and r["success"])
key = "t1_i%d" % r["init"]
src = dict(key=key, task=1, init=r["init"], seed=r["seed"], actions=arr[key])
rec = P.record_source(src)
acts, path = src["actions"], rec["path"]
ep = S.Episode(1, 0, 1000 + 100 * 1 + 0, None)
z_safe = float(path[:, 2].max())
print("DBG z_safe %.3f start hand %s" % (z_safe, ep.sim.hand().round(3)), flush=True)
for ph in S.plan(rec):
    d = np.zeros(3)
    if ph["anchor_pos"] is not None:
        d[:2] = ep.sim.positions()[ph["anchor"]][:2] - ph["anchor_pos"][:2]
    tgt = path[min(ph["pre"], len(path) - 1)] + d
    t0 = ep.t
    ep.stage(tgt, z_safe)
    err = np.linalg.norm(ep.sim.hand() - tgt)
    z0 = ep.sim.positions().get(ph["anchor"], np.zeros(3))[2]
    end = min(len(acts), ph["t"] + 5) if ph["kind"] else len(acts)
    for j in range(ph["pre"], end):
        a = np.array(acts[j], dtype=np.float64)
        a[:3] = np.clip(a[:3] + (path[min(j, len(path) - 1)] + d - ep.sim.hand()) / 0.05, -1.0, 1.0)
        ep.step(a)
    lift = ep.sim.positions().get(ph["anchor"], np.zeros(3))[2] - z0
    pos = ep.sim.positions()
    print("DBG    after contact: cream_cheese %s butter %s basket %s hand %s fingers %s" % (
        pos["cream_cheese_1"].round(3), pos["butter_1"].round(3), pos["basket_1"].round(3), ep.sim.hand().round(3),
        ep.obs["robot0_gripper_qpos"].round(3)), flush=True)
    print("DBG phase %s anchor %s d %s | staging %d steps, err %.3f m | contact to step %d, anchor dz %.3f, "
          "success %s, t %d" % (ph["kind"], ph["anchor"], (100 * d[:2]).round(1), ep.t - t0, err, end, lift,
                                ep.success, ep.t), flush=True)
pos = ep.sim.positions()
print("DBG final: cream_cheese %s butter %s basket %s" % (pos["cream_cheese_1"].round(3), pos["butter_1"].round(3),
                                                       pos["basket_1"].round(3)), flush=True)
ep.close()
