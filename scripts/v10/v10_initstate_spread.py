"""
V10 generalization test, preparation (10-22): how far do objects move between the init states of a base task?
The evaluation runs every variant from init state 0; the planned generalization test runs the camera variants from
other init states, with references recorded on init states other than the target's. For each base task of the
evaluation's 20 camera variants (one camera variant per base task builds the env; the camera does not move objects),
every init state of its init file is set in the simulator (no policy, no steps), and every movable object's position
and the robot's joint positions are compared with init state 0's.
    python v10_initstate_spread.py
"""
import collections
import functools
import json
import os


import numpy as np
import torch

# the init files are numpy pickles from the LIBERO-plus repo; lerobot's eval loads them with weights_only=False too
torch.load = functools.partial(torch.load, weights_only=False)

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

E = "/root/pace/runs/eval/"
rows = [json.loads(l) for l in open(E + "plus_l10_A_r3/episodes.jsonl")]
cam = {}
for r in rows:
    if r["category"] == "Camera Viewpoints":
        cam.setdefault(r["base"], set()).add(r["task_id"])
suite = benchmark.get_benchmark_dict()["libero_10"]()
out = {}
for base, tids in sorted(cam.items()):
    tid = min(tids)
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=64, camera_widths=64)
    env.seed(0)
    env.reset()
    inits = suite.get_task_init_states(tid)
    inner = env.env
    robot = inner.robots[0]
    jidx = robot._ref_joint_pos_indexes
    objs = list(inner.objects_dict.keys())

    def snapshot(state):
        env.set_init_state(state)
        pos = {n: inner.sim.data.body_xpos[inner.obj_body_id[n]].copy() for n in objs}
        return pos, inner.sim.data.qpos[jidx].copy()

    p0, q0 = snapshot(inits[0])
    disp = collections.defaultdict(list)  # object -> displacement (m) of init k vs init 0
    qd = []
    for k in range(1, len(inits)):
        pk, qk = snapshot(inits[k])
        for n in objs:
            disp[n].append(float(np.linalg.norm(pk[n][:2] - p0[n][:2])))  # in the table plane
        qd.append(float(np.abs(qk - q0).max()))
    env.close()
    allmax = [max(disp[n][k] for n in objs) for k in range(len(inits) - 1)]
    print(f"\n{base[:60]} (camera variants {sorted(tids)}; built from {tid}): {len(inits)} init states, "
          f"{len(objs)} objects; robot joints vs init 0: max |diff| median {np.median(qd):.4f} rad")
    print(f"  largest object shift per init state (cm): median {100*np.median(allmax):.1f}, "
          f"min {100*min(allmax):.1f}, max {100*max(allmax):.1f}")
    for n in objs:
        v = 100 * np.array(disp[n])
        print(f"    {n:28s} shift cm: median {np.median(v):5.1f}  75% {np.percentile(v, 75):5.1f}  max {v.max():5.1f}")
    out[base] = {"n_init": len(inits), "robot_qdiff_median": float(np.median(qd)),
                 "largest_shift_cm": [100 * x for x in allmax],
                 "per_object_cm": {n: [100 * x for x in disp[n]] for n in objs}}
json.dump(out, open("/root/pace/runs/rectest/v10_initstate_spread.json", "w"), indent=1)
print("-> /root/pace/runs/rectest/v10_initstate_spread.json")
