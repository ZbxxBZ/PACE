"""11-01: which task objects does pi0.5's agentview camera actually see at reset? Projects every object of the task into the
agentview image of the variant (init 1), for the 20 camera test variants, the 40 tilt-check variants and the original camera."""
import json, sys
import numpy as np
import closed_loop_v11 as C
import v10_oracle_shift2 as O
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

ROOT = C.ROOT
O.worker_init()
F = {x["task_id"]: x for x in json.load(open(f"{ROOT}/rectest/v10_shift/full_targets.json"))}
test = sorted({json.loads(l)["task_id"] for l in open(f"{ROOT}/rectest/v10_shift/base_targets/episodes.jsonl")})
tilt = [x["task_id"] for x in json.load(open(f"{ROOT}/v11/closed/tilt/targets.json"))]
src = [json.loads(l) for l in open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")]
lr5_src = [r["task_id"] for r in src if r["base"].startswith("LIVING_ROOM_SCENE5") and r["success"]][:1]
out = {}
S = 360
for t in lr5_src + test + tilt:
    fn = C.make_env(t, 1)
    env = fn.func(1, **fn.keywords)
    try:
        env.reset(seed=1000 + 7 * t + 1)
        e = env._env.env
        sim = e.sim
        M = get_camera_transform_matrix(sim, "agentview", S, S)
        names = list(e.objects_dict)
        P = np.array([sim.data.body_xpos[sim.model.body_name2id(e.objects_dict[n].root_body)] for n in names])
        px = project_points_from_world_to_camera(P, M, S, S)
        goal = list(getattr(e, "obj_of_interest", []))
        rows = {n: [int(px[i, 0]), int(px[i, 1]), bool(0 <= px[i, 0] < S and 0 <= px[i, 1] < S)] for i, n in enumerate(names)}
        out[t] = dict(name=F[t]["name"][-60:], base=F[t]["base"], goal=goal, objects=rows)
        miss = [n for n in goal if n in rows and not rows[n][2]]
        print(t, F[t]["base"][:30], F[t]["name"].split("_view_")[-1][:18], "goal", goal, "OUT of frame:", miss, flush=True)
    finally:
        env.close()
json.dump(out, open(f"{ROOT}/v11/closed/lr5/inview.json", "w"))
