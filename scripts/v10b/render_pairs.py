"""
Render the frames of V10-1b's training episodes by exact replay (10-24). Inputs: the references (collect_pi05.py --set
orig -> runs/v10b/refs, actions .npy) and the pairs (collect_pairs.py -> runs/v10b/pairs/<kind>/*.npz). Each episode is
replayed from its init state and seed under --n_aug random augmentations (augment.py; references default 2, pairs 1),
and at every policy call (each 10 steps) and gripper command change both cameras are kept at 224 x 224 with the robot
state. A pair episode is rendered under its current scene's augmentation number pair_idx (its rank among that
scene's pairs, augment.scene_aug), the one its pi0.5 label was collected under. Per (episode, augmentation) <out>/<set>/<key>_a<i>.npz: agent, wrist (uint8 [N, 224, 224, 3]), t, eef_pos,
eef_quat, joints, gripper, aug (json), success (must equal the recorded one: the replay is exact).
    python render_pairs.py --check                 # physics unchanged by augmentation + a contact sheet
    python render_pairs.py --workers 12            # -> runs/v10b/frames/{refs,expert,perturbed,negative}
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import time
import zlib

import numpy as np

import augment as A
import collect_pairs as P
import v10_oracle_shift2 as O

S = 224
CALL = 10


def make_env(tid, k, size):
    c = O.G["env_cfg"]
    fn = O.G["make_fns"](suite=O.G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=O.G["cams"],
                         episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=O.G["gk"],
                         control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    kw = dict(fn.keywords)
    kw.update(observation_width=size, observation_height=size)
    return fn.func(int(k), **kw)


def replay(tid, k, seed, actions, aug=None, size=S, frames=True):
    """Replay; returns (success, per-step hand positions, recorded calls)."""
    env = make_env(tid, k, size)
    try:
        env.reset(seed=int(seed))
        inner = env._env.env
        if aug is not None:
            A.apply(inner, aug)
        rng = np.random.default_rng(aug["seed"] if aug else 0)
        robot = inner.robots[0]
        hands, rec, ok = [], dict(agent=[], wrist=[], t=[], eef_pos=[], eef_quat=[], joints=[], gripper=[]), False
        prev_g = None
        for t, a in enumerate(actions):
            hands.append(np.array(inner.sim.data.site_xpos[robot.eef_site_id]))
            g = a[6] > 0
            if frames and (t % CALL == 0 or g != prev_g):
                o = inner._get_observations(force_update=True)
                for cam, nm in (("agentview", "agent"), ("robot0_eye_in_hand", "wrist")):
                    img = np.ascontiguousarray(o[cam + "_image"])
                    rec[nm].append(A.pixels(img, aug, rng) if aug else img)
                rec["t"].append(t)
                rec["eef_pos"].append(o["robot0_eef_pos"])
                rec["eef_quat"].append(o["robot0_eef_quat"])
                rec["joints"].append(o["robot0_joint_pos"])
                rec["gripper"].append(o["robot0_gripper_qpos"])
            prev_g = g
            _, _, _, _, info = env.step(np.asarray(a, dtype=np.float64))
            if info.get("is_success"):
                ok = True
                break
        return ok, np.stack(hands), {k_: np.stack(v) for k_, v in rec.items() if v}
    finally:
        env.close()


def job(args):
    kind, ep, i, out = args
    path = f"{out}/{kind}/{ep['key']}_a{i}.npz"
    if os.path.exists(path):
        return None
    if kind == "refs":
        aug = A.sample(np.random.default_rng([zlib.crc32(f"{kind}/{ep['key']}".encode()), i]))
    elif kind == "pi05_aug":  # pi0.5's own rollout under the scene augmentation it ran with (10-25, V10-3 value labels)
        aug = A.scene_aug(ep["task_id"], ep["init"], ep["aug_idx"])
    else:  # the scene's augmentation number pair_idx (+ i): the same one pi0.5 was labelled under
        aug = A.scene_aug(ep["task_id"], ep["init"], ep["pair_idx"] + i)
    acts = np.load(ep["file"])["actions"] if ep["file"].endswith(".npz") else np.load(ep["file"])
    ok, _, rec = replay(ep["task_id"], ep["init"], ep["seed"], acts.astype(np.float64), aug)
    np.savez_compressed(path, **rec, aug=json.dumps(aug), success=ok)
    return dict(kind=kind, key=ep["key"], aug=i, success=ok, expected=ep["success"], calls=len(rec["t"]))


def episodes(args):
    sets = {}
    refs = [json.loads(l) for l in open(f"{args.refs}/episodes.jsonl")]
    sets["refs"] = [dict(key=f"t{r['task_id']}_i{r['init']}", task_id=r["task_id"], init=r["init"], seed=r["seed"],
                         success=r["success"], file=f"{args.refs}/{r['actions']}") for r in refs if r["success"]]
    f = f"{args.pi05_aug}/episodes.jsonl"
    if os.path.exists(f):
        sets["pi05_aug"] = [dict(key=f"t{r['task_id']}_i{r['init']}_a{r['aug_idx']}", task_id=r["task_id"], init=r["init"],
                                 seed=r["seed"], aug_idx=r["aug_idx"], success=r["success"],
                                 file=f"{args.pi05_aug}/{r['actions']}") for r in map(json.loads, open(f))]
    for kind in ("expert", "perturbed", "negative", "carry"):
        f = f"{args.pairs}/{kind}.jsonl"
        if os.path.exists(f):
            rows = [json.loads(l) for l in open(f)]
            rank, seen = {}, {}
            for r in sorted(rows, key=lambda r: (r["task_id"], r["init"], r["ref_task"], r["ref_init"])):
                sc = (r["task_id"], r["init"])
                rank[r["key"]] = seen.get(sc, 0)
                seen[sc] = rank[r["key"]] + 1
            sets[kind] = [dict(r, file=f"{args.pairs}/{kind}/{r['key']}.npz", pair_idx=rank[r["key"]] % 3)
                          for r in rows]
    return sets


def check(args):
    import cv2

    P.worker_init()
    sets = episodes(args)
    ep = sets.get("expert", sets["refs"])[0]
    acts = (np.load(ep["file"])["actions"] if ep["file"].endswith(".npz") else np.load(ep["file"])).astype(np.float64)
    ok0, h0, _ = replay(ep["task_id"], ep["init"], ep["seed"], acts, None, 64, frames=False)
    tiles = []
    for i in range(4):
        aug = A.sample(np.random.default_rng([7, i]))
        ok1, h1, rec = replay(ep["task_id"], ep["init"], ep["seed"], acts, aug)
        same = h0.shape == h1.shape and np.array_equal(h0, h1)
        print(f"[check] aug {i}: success {ok0} -> {ok1}, hand path bit-identical: {same}, "
              f"max diff {np.abs(h0[:len(h1)] - h1[:len(h0)]).max():.2e}", flush=True)
        j = min(3, len(rec["t"]) - 1)
        tiles.append(np.concatenate([rec["agent"][j][::-1, ::-1], rec["wrist"][j][::-1, ::-1]], 0))
    os.makedirs(args.frames, exist_ok=True)
    cv2.imwrite(f"{args.frames}/check_aug.png", cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
    print(f"-> {args.frames}/check_aug.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="/root/pace/runs/v10b/refs")
    ap.add_argument("--pairs", default="/root/pace/runs/v10b/pairs")
    ap.add_argument("--pi05_aug", default="/root/pace/runs/v10b/pi05_aug")
    ap.add_argument("--frames", default="/root/pace/runs/v10b/frames")
    ap.add_argument("--n_aug_refs", type=int, default=2)
    ap.add_argument("--n_aug_pairs", type=int, default=1)
    ap.add_argument("--sets", nargs="+", default=["refs", "expert", "perturbed", "negative"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        return check(args)
    sets = episodes(args)
    jobs = []
    for kind in args.sets:
        os.makedirs(f"{args.frames}/{kind}", exist_ok=True)
        n = args.n_aug_refs if kind == "refs" else args.n_aug_pairs
        jobs += [(kind, ep, i, args.frames) for ep in sets.get(kind, []) for i in range(n)]
    print(f"[render] {len(jobs)} episode renders", flush=True)
    t0, bad, done = time.time(), 0, 0
    log = open(f"{args.frames}/render.jsonl", "a")
    with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            if r:
                log.write(json.dumps(r) + "\n")
                log.flush()
                done += 1
                bad += r["success"] != r["expected"]
            if (i + 1) % 100 == 0 or i + 1 == len(jobs):
                print(f"[render] {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s), replay mismatches {bad}/{done}",
                      flush=True)
    print(f"-> {args.frames}", flush=True)


if __name__ == "__main__":
    main()
