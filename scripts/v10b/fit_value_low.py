"""
The deployable V10-3 value head (10-25 20:40, user: run the V10-3 closed loop for the ablation): value_cv.py's "low"
logistic regression WITHOUT the task one-hot (the test variants' task ids do not map onto the training tasks'), fitted
on all explore + pi0.5 decision samples; cross-validated first (same folds) to pick C and the margin m. Features:
old heads P(follow), P(pi0.5), their difference, |d| / 5 cm, decision kind start / boundary, phase fraction, tau / 600.
Saved as an .npz (per output: mu, sd, w, b; plus margin) for closed_loop_v3.py --value.
    python fit_value_low.py -> runs/v10b/value/low_run1.npz
"""
import json

import numpy as np
import torch

import value_cv as VC

V = "/root/pace/runs/v10b"


def feats(X, kind, ph, tau):
    return np.c_[X[:, 0], X[:, 1], X[:, 0] - X[:, 1], X[:, 2] / 5.0, (kind == 0), (kind == 1), ph, tau].astype(np.float64)


def main():
    E = torch.load(f"{V}/value/emb_run1.pt", weights_only=False)
    X, M = E["X"], E["meta"]
    kind = np.array([m["kind"] for m in M])
    out = np.array([m["out"] for m in M])
    y = np.array([m["y"] for m in M])
    ex = np.array([m["src"] == "explore" for m in M])
    ph = np.array([m["ph_frac"] for m in M])
    tau = np.array([m["tau"] for m in M]) / 600.0
    prop = np.array([m["p"] for m in M])
    Z = feats(X, kind, ph, tau)
    sc_key = np.array([f"{m['tasks'][0]}_{m['ep'].split('_i')[1].split('_')[0]}" for m in M])
    uniq = np.unique(sc_key)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(uniq, rng.integers(0, 5, len(uniq))))
    fold = np.array([fold_of[s] for s in sc_key])
    idx = np.where(ex)[0]
    pr = np.where(out[idx] == 1, prop[idx], 1 - prop[idx])
    best = None
    for C in (0.01, 0.1, 1.0):
        P = np.full((len(M), 2), np.nan)
        for f in range(5):
            tr, te = fold != f, fold == f
            for o in (0, 1):
                m_tr = tr & (out == o) & (ex if o == 1 else True)
                P[te, o] = VC.LogisticRegression(C).fit(Z[m_tr], y[m_tr]).predict_proba(Z[te])[:, 1]
        for mth in (-0.1, -0.05, 0.0, 0.05, 0.1):
            want = (P[idx, 1] - P[idx, 0] > mth).astype(int)
            v = float(np.sum((want == out[idx]) * y[idx] / pr) / len(idx))
            print(f"[cv] C={C} m={mth:+.2f}: IPS {v:.3f}, takes {want.mean():.2f}", flush=True)
            if best is None or v > best[0]:
                best = (v, C, mth)
    always = float(np.sum((out[idx] == 1) * y[idx] / pr) / len(idx))
    print(f"[cv] best IPS {best[0]:.3f} (C={best[1]}, m={best[2]:+.2f}); always take {always:.3f}", flush=True)
    _, C, mth = best
    save = dict(margin=mth, C=C)
    for o, nm in ((0, "cont"), (1, "take")):
        m_tr = (out == o) & (ex if o == 1 else True)
        clf = VC.LogisticRegression(C).fit(Z[m_tr], y[m_tr])
        save.update({f"{nm}_mu": clf.mu, f"{nm}_sd": clf.sd, f"{nm}_w": clf.w.detach().numpy(),
                     f"{nm}_b": clf.b.detach().numpy()})
    np.savez(f"{V}/value/low_run1.npz", **save)
    json.dump(dict(best_ips=best[0], C=C, margin=mth, always_take=always), open(f"{V}/value/low_run1.json", "w"))
    print(f"-> {V}/value/low_run1.npz", flush=True)


if __name__ == "__main__":
    main()
