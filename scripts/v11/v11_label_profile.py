"""Where the V11 clear-view label (labels_cam - api, both normalized)'s energy sits: executed steps 1-10 vs steps 11-50 of the chunk, and per action dim
(normalized space, as the loss sees it). Share of the step-weighted loss at Delta = 0 (w = 3 on steps 1-10, 1 after)."""
import glob, numpy as np
V = "/root/pace/runs/v11"
nz = np.load(f"{V}/api/norm.npz"); print("norm keys", list(nz.keys()))
W = np.ones(50); W[:10] = 3; W /= W.sum()
tot = {}
for kind in ["expert", "perturbed", "carry", "pi05_aug"]:
    s2 = np.zeros((50, 7)); n = 0
    for f in glob.glob(f"{V}/labels_cam/{kind}/*.npy"):
        a = np.load(f).astype(np.float64) - np.load(f.replace("/labels_cam/", "/api/")).astype(np.float64)
        a = a[~np.isnan(a).any((1, 2))]
        s2 += (a ** 2).sum(0); n += len(a)
    ms = s2 / n  # mean square per step, dim
    tot[kind] = (ms, n)
for kind, (ms, n) in list(tot.items()) + [("all", (sum(m * k for m, k in tot.values()) / sum(k for _, k in tot.values()), sum(k for _, k in tot.values())))]:
    e = ms.mean(1)  # per step, averaged over dims
    share10 = (W[:10] * e[:10]).sum() / (W * e).sum()
    print(f"{kind:9s} n={n:6d} rms 1-10 {np.sqrt(e[:10].mean()):.3f} | 11-50 {np.sqrt(e[10:].mean()):.3f} | "
          f"step 1/10/20/30/50 {np.sqrt(e[0]):.3f}/{np.sqrt(e[9]):.3f}/{np.sqrt(e[19]):.3f}/{np.sqrt(e[29]):.3f}/{np.sqrt(e[49]):.3f} | "
          f"loss share of steps 1-10: {share10:.0%}")
    d10 = ms[:10].mean(0)
    print(f"{'':9s} steps 1-10 per dim rms: pos {np.sqrt(d10[:3].mean()):.3f} rot {np.sqrt(d10[3:6].mean()):.3f} grip {np.sqrt(d10[6]):.3f}; "
          f"energy share pos/rot/grip {d10[:3].sum() / d10.sum():.0%}/{d10[3:6].sum() / d10.sum():.0%}/{d10[6] / d10.sum():.0%}")
