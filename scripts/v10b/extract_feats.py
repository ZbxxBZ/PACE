"""
Frozen SigLIP features of the V10-1b frames (10-24). pi0.5's own vision tower (as v10_loc_probe.py: pixels in [-1, 1],
224 x 224, 16 x 16 patch tokens of width 1152), projected to --dim by a PCA fitted on a sample of tokens (all 274k
frames x 256 tokens x 1152 in fp16 would be ~160 GB; PCA-384 keeps the patch grid at ~1/3 of that). The PCA is the
same for every frame and camera and is saved, so the closed loop applies the identical projection.
Per frame file <frames>/<set>/<key>_a<i>.npz -> <out>/<set>/<key>_a<i>.npy, fp16 [N calls, 2 cams (agent, wrist),
256, dim].
    python extract_feats.py -> runs/v10b/feats (pca.npz, <set>/*.npy)
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

SETS = ("refs", "expert", "perturbed", "negative")


def load_tower():
    from pace.common import load_policy

    policy, _ = load_policy("lerobot/pi05_libero_finetuned", "cuda", n_action_steps=10)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower.eval()
    del policy.model.paligemma_with_expert.paligemma.model.language_model  # only the vision tower is needed
    torch.cuda.empty_cache()
    return vt


@torch.no_grad()
def encode(vt, imgs, bs=64):
    """uint8 [N, 224, 224, 3] -> float32 [N, 256, 1152] on the GPU."""
    dt = next(vt.parameters()).dtype
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1.0
        out.append(vt(pixel_values=x.to(dt)).last_hidden_state.float())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/root/pace/runs/v10b/frames")
    ap.add_argument("--out", default="/root/pace/runs/v10b/feats")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--fit_files", type=int, default=300)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    args = ap.parse_args()
    sets = args.sets
    files = {s: sorted(glob.glob(f"{args.frames}/{s}/*.npz")) for s in sets}
    vt = load_tower()
    t0 = time.time()
    pca_path = f"{args.out}/pca.npz"
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(pca_path):
        z = np.load(pca_path)
        mean, comp = torch.from_numpy(z["mean"]).cuda(), torch.from_numpy(z["comp"]).cuda()
    else:
        rng = np.random.default_rng(0)
        pool = [f for s in sets for f in files[s]]
        toks = []
        for f in rng.choice(pool, min(args.fit_files, len(pool)), replace=False):
            z = np.load(f)
            j = rng.choice(len(z["t"]), min(4, len(z["t"])), replace=False)
            for cam in ("agent", "wrist"):
                h = encode(vt, z[cam][j])
                toks.append(h.reshape(-1, h.shape[-1])[torch.randperm(h.shape[0] * h.shape[1])[:256]])
        X = torch.cat(toks).double()
        mean = X.mean(0)
        evals, evecs = torch.linalg.eigh((X - mean).T @ (X - mean) / len(X))  # ascending
        var = evals.flip(0) / evals.sum()
        comp = evecs.flip(1).T[:args.dim].float()
        mean = mean.float()
        np.savez(pca_path, mean=mean.cpu().numpy(), comp=comp.cpu().numpy(), var=var[:args.dim].cpu().numpy())
        print(f"[pca] {X.shape[0]} tokens, {args.dim} dims keep {100 * var[:args.dim].sum():.1f}% of the variance "
              f"({time.time() - t0:.0f}s)", flush=True)
    n = 0
    for s in sets:
        os.makedirs(f"{args.out}/{s}", exist_ok=True)
        for f in files[s]:
            dst = f"{args.out}/{s}/{os.path.basename(f)[:-4]}.npy"
            if os.path.exists(dst):
                continue
            z = np.load(f)
            cams = [((encode(vt, z[c]) - mean) @ comp.T).half() for c in ("agent", "wrist")]
            np.save(dst, torch.stack(cams, 1).cpu().numpy())
            n += 1
            if n % 200 == 0:
                print(f"[feats] {n} files ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {args.out} ({n} files, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
