"""
V11b training data from exact replays of V11's own episodes (11-01; desktop design doc 10.14f). Episodes are drawn at random per
kind (expert, perturbed, carry, pi05_aug; seed 11), disjoint between the modes:
  keep : camera move removed, lights, colours and pixel noise kept. Delta* = 0 by the label's definition; a_pi is V11's
         clear-view chunk (same frames, same noise seed), copied.
  pure : no augmentation at all. Delta* = 0; a_pi computed here.
  wide : the camera changed as LIBERO-plus changes it (orbit, farther away, turned in place; augment.sample_wide, our own values,
         no LIBERO-plus variant), lights etc. kept. Delta* = V11's clear-view chunk - a_pi, so that chunk is copied as the label.
Same file formats and row indices as V11's data: <out>/frames/<mode>/<fkey>.npz, <out>/api/<mode>/<fkey>.npy,
<out>/labels/wide/<fkey>.npy, <out>/<mode>.jsonl. fkey = <kind>__<key>_a0. Resumable.
    python v11b_replay.py --mode keep --out runs/v11b
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
import zlib

import numpy as np
import torch

import augment as A
import render_pairs as R
from v11_labels import V, V11, episodes

KINDS = ("expert", "perturbed", "carry", "pi05_aug")
RANGES = dict(keep=(0.0, 0.125), pure=(0.125, 0.25), wide=(0.25, 0.65))
CALL = 10


def render(job):
    mode, kind, key, tid, k, seed, afile, aug = job
    if afile.endswith(".npz"):
        with np.load(afile) as z:
            acts = np.array(z["actions"], np.float64)
    else:
        acts = np.load(afile).astype(np.float64)
    ok, _, rec = R.replay(tid, k, seed, acts, aug)
    return kind, key, tid, k, seed, aug, ok, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(RANGES), required=True)
    ap.add_argument("--out", default="/root/pace/runs/v11b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first episodes only")
    args = ap.parse_args()
    m = args.mode
    for d in (f"frames/{m}", f"api/{m}") + (("labels/wide",) if m == "wide" else ()):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    import collect_pairs as P
    from lerobot.envs.utils import preprocess_observation
    from v11_check import load_pi05, raw_obs

    lang = {r["task_id"]: r["language"] for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))}
    jobs = []
    lo, hi = RANGES[m]
    for kind in KINDS:
        eps = [e for e in episodes(kind) if e[5] and os.path.exists(f"{V}/frames/{kind}/{e[0]}_a0.npz")
               and os.path.exists(f"{V11}/labels_cam/{kind}/{e[0]}_a0.npy")]
        eps = [eps[i] for i in np.random.default_rng(11).permutation(len(eps))]
        chosen = eps[int(lo * len(eps)):int(hi * len(eps))]
        print(f"[replay] {m} {kind}: {len(chosen)} of {len(eps)} episodes", flush=True)
        for key, tid, k, seed, afile, _ in chosen:
            if os.path.exists(f"{args.out}/api/{m}/{kind}__{key}_a0.npy"):
                continue
            with np.load(f"{V}/frames/{kind}/{key}_a0.npz") as z:
                aug = json.loads(str(z["aug"]))
            if m == "pure":
                aug = None
            else:
                aug = dict(aug, cam_dpos=[0.0, 0.0, 0.0], cam_angle=0.0)
                if m == "wide":
                    aug["cam_wide"] = A.sample_wide(np.random.default_rng([zlib.crc32(f"{kind}/{key}".encode()), 7]))
            jobs.append((m, kind, key, tid, k, seed, afile, aug))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[replay] {m}: {len(jobs)} episodes to do", flush=True)
    policy, noise, pre, env_pre = (None, None, None, None) if m == "keep" else load_pi05()[:4]
    log = open(f"{args.out}/{m}.jsonl", "a")
    t0, bad, samples = time.time(), 0, 0
    with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:
        for n, (kind, key, tid, k, seed, aug, ok, rec) in enumerate(pool.imap_unordered(render, jobs)):
            with np.load(f"{V}/frames/{kind}/{key}_a0.npz") as z:
                stored_ok, stored_t = bool(z["success"]), np.array(z["t"])
            if not (ok == stored_ok and np.array_equal(rec["t"], stored_t)):
                bad += 1
                print(f"[replay] {kind} {key}: replay differs (success {ok} vs {stored_ok}), skipped", flush=True)
                continue
            fkey = f"{kind}__{key}_a0"
            calls = [i for i, t in enumerate(rec["t"]) if int(t) % CALL == 0]
            clear = f"{V11}/labels_cam/{kind}/{key}_a0.npy"
            np.savez_compressed(f"{args.out}/frames/{m}/{fkey}.npz", **rec, aug=json.dumps(aug), success=ok)
            if m == "keep":
                shutil.copy(clear, f"{args.out}/api/{m}/{fkey}.npy")
            else:
                a_pi = np.full((len(rec["t"]), 50, 7), np.nan, np.float32)
                for b0 in range(0, len(calls), args.batch):
                    rows = calls[b0:b0 + args.batch]
                    obs = preprocess_observation(raw_obs([(rec, i) for i in rows]))
                    obs["task"] = [lang[tid]] * len(rows)
                    obs = pre(env_pre(obs))
                    noise.set([zlib.crc32(f"{key}/{i}".encode()) for i in rows])
                    with torch.inference_mode():
                        a_pi[rows] = policy.predict_action_chunk(obs).float().cpu().numpy()
                np.save(f"{args.out}/api/{m}/{fkey}.npy", a_pi.astype(np.float16))
                if m == "wide":
                    shutil.copy(clear, f"{args.out}/labels/wide/{fkey}.npy")
            samples += len(calls)
            log.write(json.dumps(dict(mode=m, kind=kind, key=key, fkey=fkey, task_id=tid, init=k, seed=seed, calls=len(calls),
                                      cam_wide=aug.get("cam_wide") if aug else None)) + "\n")
            log.flush()
            if n % 50 == 0:
                print(f"[replay] {m} {n + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {args.out} {m}: {samples} samples, {bad} skipped", flush=True)
    assert bad <= 0.01 * max(len(jobs), 1)


if __name__ == "__main__":
    main()
