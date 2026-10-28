"""
How much of run1's d comes from the images? (10-24) The best answer that uses only the state input: d = A @ offset + b
(offset = current hand - reference hand at the aligned step, xy), A, b fitted by least squares on the TRAINING samples
(expert + perturbed in their natural mix), scored on validation exactly like train_v10b.evaluate (median / 75th /
90th by the reference hand's distance to the contact point). Also the per-bin fit (A, b per distance bin).
    python v10b_blind.py runs/v10b/train/index_feats.pt
"""
import sys

import numpy as np
import torch

BINS = [(0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 1.0)]


def rows(idx, split):
    X, Y, D, K = [], [], [], []
    for s in idx["samples"]:
        ep = idx["eps"][s["ep"]]
        if ep["split"] != split or s["d"] is None:
            continue
        rm = idx["refs"][ep["ref"]][min(idx["refs"][ep["ref"]])]
        j = int(np.argmin(np.abs(rm["t"] - s["t"])))
        X.append(ep["meta"]["eef"][s["c"]][:2] - rm["eef"][j][:2])
        Y.append(s["d"])
        D.append(s["dist"])
        K.append(ep["kind"])
    return np.array(X), np.array(Y), np.array(D), np.array(K)


def fit(X, Y):
    Xa = np.c_[X, np.ones(len(X))]
    W, *_ = np.linalg.lstsq(Xa, Y, rcond=None)
    return W


def main():
    idx = torch.load(sys.argv[1], weights_only=False)
    Xt, Yt, Dt, _ = rows(idx, "train")
    Xv, Yv, Dv, Kv = rows(idx, "val")
    W = fit(Xt, Yt)
    print("global fit A, b:", np.round(W, 3).tolist())
    Wb = [fit(Xt[(Dt >= lo) & (Dt < hi)], Yt[(Dt >= lo) & (Dt < hi)]) for lo, hi in BINS]
    for kind in ("perturbed", "expert"):
        print(f"-- {kind}: median / 75th / 90th (cm): global linear | per-bin linear")
        for (lo, hi), wb in zip(BINS, Wb):
            sel = (Kv == kind) & (Dv >= lo) & (Dv < hi)
            Xa = np.c_[Xv[sel], np.ones(sel.sum())]
            e1 = 100 * np.linalg.norm(Xa @ W - Yv[sel], axis=1)
            e2 = 100 * np.linalg.norm(Xa @ wb - Yv[sel], axis=1)
            q = lambda e: " / ".join(f"{np.percentile(e, p):.2f}" for p in (50, 75, 90))
            print(f"  {100 * lo:3.0f}-{100 * hi:<3.0f} cm: {q(e1)} | {q(e2)}  n={sel.sum()}")


if __name__ == "__main__":
    main()
