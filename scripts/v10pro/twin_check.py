"""Visual check of the twin (10-23): real swapped scene vs. twin, both cameras, at the first query of a few tasks."""
import imageio
import numpy as np

import pro_common as C
import pro_oracle as P
import pro_twin as T

rows = []
for t, anchor in ((7, "alphabet_soup_1"), (4, "porcelain_mug_1"), (9, "white_yellow_mug_1")):
    seed = 1000 + 100 * t
    env, task, inits = C.make_env("libero_10_swap", t, C.SIZE)
    twin = T.Twin(task.name, 0, seed)
    obs = C.reset(env, inits[0], seed)
    d = np.zeros(3)
    d[:2] = P.Sim(env).positions()[anchor][:2] - twin.known[anchor][:2]
    err = twin.sync(env, d)
    tobs = twin.observation()
    flip = lambda x: x[::-1, ::-1]
    rows.append(np.concatenate([flip(obs["agentview_image"]), flip(obs["robot0_eye_in_hand_image"]),
                                flip(tobs["agentview_image"]), flip(tobs["robot0_eye_in_hand_image"])], axis=1))
    print("CHK task %d anchor %s d %s cm, ik err %.4f m, real eef %s twin eef %s" % (
        t, anchor, (100 * d[:2]).round(1), err, obs["robot0_eef_pos"].round(3), tobs["robot0_eef_pos"].round(3)),
        flush=True)
    twin.close()
    env.close()
imageio.imwrite("/tmp/twin_check.png", np.concatenate(rows, axis=0)[::2, ::2])
