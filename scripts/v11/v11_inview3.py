"""11-01: which task objects are inside pi0.5's agentview at reset, for all 419 camera variants (init 1) plus the original
camera of each base task. Pixel positions are computed without clipping (robosuite's projection clips to the image, so 359
means outside). Fixtures (cabinet, microwave, stove) are included."""
import json
import numpy as np
import closed_loop_v11 as C
import v10_oracle_shift2 as O
from robosuite.utils.camera_utils import get_camera_transform_matrix

ROOT = C.ROOT
O.worker_init()
F = json.load(open(f"{ROOT}/rectest/v10_shift/full_targets.json"))
cam = [x for x in F if x["category"] == "Camera Viewpoints"]
light = {}
for x in F:  # one original-camera variant per base task
    if x["category"] == "Light Conditions":
        light.setdefault(x["base"], x)
S = 360
out = {}
for x in list(light.values()) + cam:
    t = x["task_id"]
    fn = C.make_env(t, 1)
    env = fn.func(1, **fn.keywords)
    try:
        env.reset(seed=1000 + 7 * t + 1)
        e = env._env.env
        sim = e.sim
        M = get_camera_transform_matrix(sim, "agentview", S, S)
        goal = list(getattr(e, "obj_of_interest", []))
        rows = {}
        for n in sorted(set(goal) | set(e.objects_dict)):
            o = e.objects_dict.get(n) or e.fixtures_dict.get(n)
            if o is None:
                continue
            p = M @ np.r_[sim.data.body_xpos[sim.model.body_name2id(o.root_body)], 1.0]
            rows[n] = [round(float(p[0] / p[2]), 1), round(float(p[1] / p[2]), 1)]
        out[t] = dict(name=x["name"], base=x["base"], category=x["category"], goal=goal, px=rows)
    finally:
        env.close()
    if len(out) % 50 == 0:
        print(len(out), flush=True)
json.dump(out, open(f"{ROOT}/v11/closed/lr5/inview_all.json", "w"))
print("-> done", len(out), flush=True)
