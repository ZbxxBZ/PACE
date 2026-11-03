"""
References with the agentview camera move removed (11-01, V11b; desktop design doc 10.14f): the test's references have the
camera in place, the training ones (runs/v11/refvis) had it moved. Lights, colours and pixel noise of each reference's
augmentation 0 are kept (the test's references come from light variants); the wrist frames and states are the stored ones.
Same format as v11_check.py --refvis. Resumable.
    python v11_refvis_orig.py --out runs/v11/refvis_orig
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from v11_labels import render_job

V = "/root/pace/runs/v10b"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/pace/runs/v11/refvis_orig")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import collect_pairs as P
    from lerobot.envs.utils import preprocess_observation
    from pace.pi05 import batch_state, vis_tokens
    from v11_check import load_pi05, raw_obs

    rows = {f"t{r['task_id']}_i{r['init']}": r for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))
            if r["success"] and os.path.exists(f"{V}/frames/refs/t{r['task_id']}_i{r['init']}_a0.npz")}
    jobs = []
    for key, r in rows.items():
        if not os.path.exists(f"{args.out}/{key}_a0.npz"):
            with np.load(f"{V}/frames/refs/{key}_a0.npz") as z:
                jobs.append(("refs", key, r["task_id"], r["init"], r["seed"], f"{V}/refs/{r['actions']}", json.loads(str(z["aug"]))))
    print(f"[refvis_orig] {len(rows)} references, {len(jobs)} to do", flush=True)
    policy, _, pre, env_pre, _, _ = load_pi05()
    t0, bad = time.time(), 0
    with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:
        for n, (_, key, rec) in enumerate(pool.imap_unordered(render_job, jobs)):
            r = rows[key]
            fr = dict(np.load(f"{V}/frames/refs/{key}_a0.npz"))
            if not (rec["success"] and np.array_equal(rec["t"], fr["t"])):
                bad += 1
                print(f"[refvis_orig] {key}: replay differs (success {rec['success']}), skipped", flush=True)
                continue
            clear = dict(fr, agent=rec["agent"])
            vis, st = [], []
            for b0 in range(0, len(fr["t"]), args.batch):
                items = [(clear, i) for i in range(b0, min(len(fr["t"]), b0 + args.batch))]
                obs = preprocess_observation(raw_obs(items))
                obs["task"] = [r["language"]] * len(items)
                obs = pre(env_pre(obs))
                with torch.inference_mode():
                    images, img_masks = policy._preprocess_images(obs)
                    embs = policy.model._embed_images(images, img_masks)
                    vis.append(vis_tokens(embs, (0, 1)).half().cpu().numpy())
                    st.append(batch_state(obs).cpu().numpy())
            np.savez(f"{args.out}/{key}_a0.npz", vis=np.concatenate(vis), state=np.concatenate(st), t=fr["t"])
            if n % 20 == 0:
                print(f"[refvis_orig] {n + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {args.out} ({bad} skipped)", flush=True)
    assert bad == 0


if __name__ == "__main__":
    main()
