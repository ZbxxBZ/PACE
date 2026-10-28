"""
Control for step 1 (10-23): replay the exported experiences on the ORIGINAL libero_10 bddl / init files with whichever
LIBERO code base is active (LIBERO-plus in .venv_plus, or LIBERO-PRO in .venv_pro), same reset as lerobot's LiberoEnv
(seed -> reset -> set_init_state -> 10 no-op settle steps), success from the env's own check. Separates "the variant
scenes differ physically from the original" from "the two code bases differ". EGL on the GPU.
    python replay_original_bddl.py --tag plus|pro --workers 12
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

SRC = "/root/pace/runs/rectest/v10_pro/sources_l10"


def paths(variant):
    """The original libero_10 task a variant belongs to: (bddl path, init states)."""
    import torch
    from libero.libero import get_libero_path

    idir = os.path.join(get_libero_path("init_states"), "libero_10")
    stems = [f[:-len(".pruned_init")] for f in os.listdir(idir) if f.endswith(".pruned_init")]
    stem = max((s for s in stems if variant.startswith(s)), key=len)
    inits = torch.load(os.path.join(idir, stem + ".pruned_init"), weights_only=False) \
        if "weights_only" in torch.load.__code__.co_varnames else torch.load(os.path.join(idir, stem + ".pruned_init"))
    return os.path.join(get_libero_path("bddl_files"), "libero_10", stem + ".bddl"), np.asarray(inits)


def replay(job):
    from libero.libero.envs import OffScreenRenderEnv

    key, variant, seed, init_index, acts = job
    bddl, inits = paths(variant)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128, control_freq=20)
    try:
        env.reset()
        env.seed(seed)
        env.reset()
        env.set_init_state(inits[init_index])
        for _ in range(10):
            env.step([0, 0, 0, 0, 0, 0, -1])
        for t, a in enumerate(acts):
            env.step(a)
            if env.check_success():
                return key, True, t + 1
        return key, False, len(acts)
    finally:
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    meta = json.load(open(SRC + ".json"))
    arrays = np.load(SRC + ".npz")
    jobs = [(m["key"], m["variant"], m["seed"], m["init_index"], arrays[m["key"]]) for m in meta]
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        res = {k: (ok, n) for k, ok, n in pool.imap_unordered(replay, jobs)}
    rows = [dict(m, success=res[m["key"]][0], replay_steps=res[m["key"]][1]) for m in meta]
    out = f"/root/pace/runs/rectest/v10_pro/replay_original_{args.tag}.json"
    json.dump(rows, open(out, "w"), indent=1)
    print(f"[{args.tag}] original bddl: {sum(r['success'] for r in rows)}/{len(rows)} succeed ({time.time() - t0:.0f}s)")
    print("  failed:", sorted(r["key"] for r in rows if not r["success"]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
