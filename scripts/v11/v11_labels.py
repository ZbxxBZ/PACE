"""
V11 training labels, "clear view" version (10-27; decided after the label pilot, v11_clean_pilot.py, desktop design doc
10.14c). For every successful expert pair (train + validation) and every pi0.5 call row (t % 10 == 0) of its stored
frames (runs/v10b/frames/expert/<key>_a0.npz, under the scene augmentation): pi0.5's chunk on the stored, perturbed
frames -> a_pi (runs/v11/api/expert/<key>_a0.npy, the branch's input), and on a replay with the agentview camera move
removed (lights, colours, pixel noise unchanged) -> a_clear (runs/v11/labels_cam/expert/<key>_a0.npy). Both with the
SAME noise seed per (episode, row), zlib.crc32(f"{key}/{row}") as v11_check.py --dump, so the label
Delta* = a_clear - a_pi holds only what the camera move changes. [frame rows, 50, 7] float16, normalized, NaN on rows
that are not calls. Resumable. Negative pairs need only a_pi (their label is 0): v11_check.py --dump negative.
    python v11_labels.py --workers 8
--kinds perturbed carry pi05_aug (10-27, user: enlarge the training set): the clear-view label needs no correct action
a*, so any rendered state can be labelled -- V10's perturbed pairs (hand pushed off by 1-3 cm), carry pairs (entering a
phase with the previous phase's d) and pi0.5's own rollouts under the augmentations (pi05_aug, failures included: the
states pi0.5 itself reaches). Replays are checked to end as the stored ones did (success flag, frame rows).
    python v11_labels.py --kinds perturbed carry pi05_aug --workers 8
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import zlib

import numpy as np
import torch

V = "/root/pace/runs/v10b"
V11 = "/root/pace/runs/v11"
CALL = 10


def render_job(job):
    """(kind, key, task, init, seed, actions file, aug) -> kind, key, the replay without the agentview camera move."""
    import render_pairs as R

    kind, key, tid, k, seed, afile, aug = job
    if afile.endswith(".npz"):
        with np.load(afile) as z:
            acts = np.array(z["actions"], np.float64)
    else:
        acts = np.load(afile).astype(np.float64)
    ok, _, rec = R.replay(tid, k, seed, acts, dict(aug, cam_dpos=[0.0, 0.0, 0.0], cam_angle=0.0))
    return kind, key, dict(rec, success=ok)


def episodes(kind):
    """(key, task, init, env seed, actions file) of the rendered episodes of a kind."""
    if kind == "pi05_aug":
        return [(f"t{r['task_id']}_i{r['init']}_a{r['aug_idx']}", r["task_id"], r["init"], r["seed"],
                 f"{V}/pi05_aug/{r['actions']}", True) for r in map(json.loads, open(f"{V}/pi05_aug/episodes.jsonl"))]
    return [(r["key"], r["task_id"], r["init"], r["seed"], f"{V}/pairs/{kind}/{r['key']}.npz",
             kind != "expert" or r["success"]) for r in map(json.loads, open(f"{V}/pairs/{kind}.jsonl"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16, help="samples per forward (x 2 rows)")
    ap.add_argument("--kinds", nargs="+", default=["expert"], help="expert (successful only) / perturbed / carry / pi05_aug")
    args = ap.parse_args()
    import collect_pairs as P
    from lerobot.envs.utils import preprocess_observation
    from v11_check import load_pi05, raw_obs

    lang = {r["task_id"]: r["language"] for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))}
    jobs, task_of = [], {}
    for kind in args.kinds:
        os.makedirs(f"{V11}/api/{kind}", exist_ok=True)
        os.makedirs(f"{V11}/labels_cam/{kind}", exist_ok=True)
        n0 = len(jobs)
        for key, tid, k, seed, afile, use in episodes(kind):
            f = f"{V}/frames/{kind}/{key}_a0.npz"
            if not (use and os.path.exists(f)):
                continue
            if os.path.exists(f"{V11}/api/{kind}/{key}_a0.npy") and os.path.exists(f"{V11}/labels_cam/{kind}/{key}_a0.npy"):
                continue
            with np.load(f) as z:
                jobs.append((kind, key, tid, k, seed, afile, json.loads(str(z["aug"]))))
            task_of[(kind, key)] = tid
        print(f"[labels] {kind}: {len(jobs) - n0} episodes to do", flush=True)
    policy, noise, pre, env_pre, mean, std = load_pi05()
    np.savez(f"{V11}/api/norm.npz", mean=mean, std=std)
    t0, bad = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:
        for n, (kind, key, rec) in enumerate(pool.imap_unordered(render_job, jobs)):
            with np.load(f"{V}/frames/{kind}/{key}_a0.npz") as z:
                orig = {k: np.array(z[k]) for k in ("agent", "wrist", "eef_pos", "eef_quat", "gripper", "joints", "t")}
                stored_ok = bool(z["success"])
            if not (rec["success"] == stored_ok and np.array_equal(rec["t"], orig["t"])):
                bad += 1
                print(f"[labels] {kind} {key}: replay differs (success {rec['success']} vs {stored_ok}), skipped", flush=True)
                continue
            out_api, out_lab = f"{V11}/api/{kind}", f"{V11}/labels_cam/{kind}"
            clear = dict(orig, agent=rec["agent"])
            calls = [i for i, t in enumerate(orig["t"]) if int(t) % CALL == 0]
            a_pi = np.full((len(orig["t"]), 50, 7), np.nan, np.float32)
            a_cl = np.full_like(a_pi, np.nan)
            for b0 in range(0, len(calls), args.batch):
                rows = calls[b0:b0 + args.batch]
                obs = preprocess_observation(raw_obs([(orig, i) for i in rows] + [(clear, i) for i in rows]))
                obs["task"] = [lang[task_of[(kind, key)]]] * (2 * len(rows))
                obs = pre(env_pre(obs))
                seeds = [zlib.crc32(f"{key}/{i}".encode()) for i in rows]
                noise.set(seeds + seeds)
                with torch.inference_mode():
                    a = policy.predict_action_chunk(obs).float().cpu().numpy()
                a_pi[rows], a_cl[rows] = a[:len(rows)], a[len(rows):]
            np.save(f"{out_api}/{key}_a0.npy", a_pi.astype(np.float16))
            np.save(f"{out_lab}/{key}_a0.npy", a_cl.astype(np.float16))
            if n % 50 == 0:
                print(f"[labels] {n + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {V11}/api, {V11}/labels_cam ({bad} skipped)", flush=True)


if __name__ == "__main__":
    main()
