"""Mark the projected objects on the actual agentview frames (what pi0.5 receives) and on lerobot's render frame."""
import json
import numpy as np, cv2, imageio.v2 as imageio
import closed_loop_v11 as C
import v10_oracle_shift2 as O
from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
ROOT = C.ROOT
O.worker_init()
S = 360
tiles = []
for t in (2433, 838, 840, 843):
    fn = C.make_env(t, 1)
    env = fn.func(1, **fn.keywords)
    try:
        obs, _ = env.reset(seed=1000 + 7 * t + 1)
        e = env._env.env
        sim = e.sim
        print(t, "obs keys", list(obs.keys()), {k: list(v.keys()) if isinstance(v, dict) else np.shape(v) for k, v in obs.items()})
        img = obs["pixels"]["image"] if "image" in obs.get("pixels", {}) else None
        names = list(e.objects_dict)
        P = np.array([sim.data.body_xpos[sim.model.body_name2id(e.objects_dict[n].root_body)] for n in names])
        M = get_camera_transform_matrix(sim, "agentview", S, S)
        px = project_points_from_world_to_camera(P, M, S, S)
        cid = sim.model.camera_name2id("agentview")
        print("  cam pos", sim.data.cam_xpos[cid].round(3).tolist(), "fovy", float(sim.model.cam_fovy[cid]), "world xyz", {n: P[i].round(3).tolist() for i, n in enumerate(names)})
        print("  row,col", {n: px[i].tolist() for i, n in enumerate(names)})
        rend = np.ascontiguousarray(np.asarray(env.render())).copy()
        print("  render", rend.shape, "pixels", None if img is None else np.shape(img))
        for nm, im in (("obs", img), ("render", rend)):
            if im is None:
                continue
            im = np.ascontiguousarray(np.asarray(im)).copy()
            h, w = im.shape[:2]
            for flip in (False, True):
                v = im.copy()
                for i, n in enumerate(names):
                    r, c = px[i] * h / S
                    r2 = h - 1 - r if flip else r
                    c2 = w - 1 - c if flip else c
                    cv2.circle(v, (int(c2), int(r2)), 6, (0, 255, 0), 2)
                    cv2.putText(v, n[:9], (int(c2) + 7, int(r2)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(v, f"{t} {nm} rot180={flip}", (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                tiles.append(cv2.resize(v, (360, 360)))
    finally:
        env.close()
n = len(tiles) // 4
imageio.imwrite(f"{ROOT}/v11/closed/lr5/inview_mark.png", np.concatenate([np.concatenate(tiles[i * n:(i + 1) * n], 1) for i in range(4)], 0))
