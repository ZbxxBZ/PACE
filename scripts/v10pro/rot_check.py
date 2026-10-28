"""How much does the experience's gripper orientation change before each contact? (stage-test diagnosis, 10-23)"""
import json

import numpy as np

import pro_common as C
import pro_oracle as P
import pro_stage as S

rows = [json.loads(l) for l in open(P.EXP + "/episodes.jsonl")]
arr = np.load(P.EXP + "/actions.npz")


def angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


for task in (1, 4, 2):
    r = next(r for r in sorted(rows, key=lambda r: r["init"]) if r["task"] == task and r["success"])
    key = "t%d_i%d" % (task, r["init"])
    acts = arr[key]
    env, _, inits = C.make_env("libero_10", task, 64)
    C.reset(env, inits[r["init"]], r["seed"])
    sim = P.Sim(env)
    Rs = []
    for a in acts:
        Rs.append(sim.sim.data.site_xmat[sim.robot.eef_site_id].reshape(3, 3).copy())
        env.step(a)
    env.close()
    src = dict(key=key, task=task, init=r["init"], seed=r["seed"], actions=acts)
    src["rec"] = P.record_source(src)
    print("ROT task", task, [(p["kind"], p["pre"], round(angle(Rs[0], Rs[min(p["pre"], len(Rs) - 1)]), 1))
                             for p in S.plan(src["rec"])], flush=True)
