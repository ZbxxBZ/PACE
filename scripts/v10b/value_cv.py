"""
V10-3 value head, second try (10-25 20:10): the first (train_value.py: 2048-d pooled queries -> MLP) overfit from epoch
0 (val BCE 0.83 -> 2.4; P(take) AUROC 0.42 on 59 validation decisions) -- ~470 "take" decisions cannot fit 2048 inputs.
Here: low-dimensional inputs and 5-fold cross-validation by scene over ALL explore decisions (883) so the estimate is
not 107 decisions.
  features "low"   : old heads P(follow), P(pi0.5), their difference, |d| (cm), decision kind (start / boundary), phase
                     fraction, tau / 600, task id one-hot (10);
  features "low+h" : + the pooled queries projected to 16 dims (PCA fitted on the training folds);
  model            : logistic regression per output (cont, take), L2, class-balanced (torch LBFGS; no sklearn here).
P(cont) also learns from pi0.5's own rollouts (pi05 samples, outcome of pi0.5 alone), P(take) only from explore "take".
Report per fold-pooled estimate: AUROC of each output on held-out explore decisions; the inverse-propensity estimate
of the success of "take iff P(take) - P(cont) > m" (m grid) vs always take / never take / logged random / the old heads
(take iff P(follow) - P(pi0.5) > 0).
Embeddings are cached by train_value.py's embed (runs/v10b/value/emb_<backbone>.pt).
    python value_cv.py --backbone runs/v10b/train/run1/best.pt
"""
import argparse
import json
import os

import numpy as np
import torch

import train_v10b as T
import train_value as TV

V = "/root/pace/runs/v10b"


class PCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mu = X.mean(0)
        _, _, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        self.W = Vt[:self.k].T
        return self

    def transform(self, X):
        return (X - self.mu) @ self.W


class LogisticRegression:
    """L2-regularised, class-balanced logistic regression (inputs standardised on the training data)."""

    def __init__(self, C=1.0):
        self.C = C

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64)
        yt = torch.tensor(y, dtype=torch.float64)
        pos = max(float(yt.mean()), 1e-3)
        w = torch.where(yt > 0.5, 0.5 / pos, 0.5 / max(1 - pos, 1e-3))
        self.w = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        self.b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=500, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xt @ self.w + self.b
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none") * w).mean()                 + 0.5 / (self.C * len(yt)) * (self.w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return self

    def predict_proba(self, X):
        z = torch.tensor((X - self.mu) / self.sd, dtype=torch.float64) @ self.w.detach() + self.b.detach()
        p = torch.sigmoid(z).numpy()
        return np.c_[1 - p, p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=f"{V}/train/run1/best.pt")
    ap.add_argument("--explore", default=f"{V}/closed/explore")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    args = ap.parse_args()

    tag = os.path.basename(os.path.dirname(args.backbone))
    cache = f"{V}/value/emb_{tag}.pt"
    ck = torch.load(args.backbone, weights_only=False)
    a = ck["args"]
    args.state = a.get("state", "abs")
    if os.path.exists(cache):
        E = torch.load(cache, weights_only=False)
    else:
        D = np.load(f"{V}/feats/pca.npz")["comp"].shape[0]
        model = T.Comparator(D, w=a["width"], pool=a["pool"]).cuda().eval()
        model.load_state_dict(ck["model"])
        refs = TV.load_refs()
        from pace.pi05 import InstructionText

        text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned",
                                               tasks=list(dict.fromkeys(r["lang"] for r in refs.values())))
        lang_vec = {r["task"]: text.embed(r["lang"]).float() for r in refs.values()}
        S = TV.collect(args, refs)
        H, X = TV.embed(model, S, refs, lang_vec, a["d_scale"])
        meta = [{k: v for k, v in x.items() if k not in ("cur", "rows", "state")} for x in S]
        E = dict(H=H.numpy(), X=X.numpy(), meta=meta)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(E, cache)
    H, X, M = E["H"], E["X"], E["meta"]
    n = len(M)
    kind = np.array([m["kind"] for m in M])
    out = np.array([m["out"] for m in M])
    y = np.array([m["y"] for m in M])
    src = np.array([m["src"] for m in M])
    task = np.array([m["tasks"][0] for m in M])
    tasks = sorted(set(task))
    ph = np.array([m["ph_frac"] for m in M])
    tau = np.array([m["tau"] for m in M]) / 600.0
    prop = np.array([m["p"] for m in M])
    scene = np.array([m["ep"].split("_t")[1] if m["src"] == "explore" else m["ep"] for m in M])
    low = np.c_[X[:, 0], X[:, 1], X[:, 0] - X[:, 1], X[:, 2] / 5.0, (kind == 0), (kind == 1), ph, tau,
                np.stack([(task == t) for t in tasks], 1)].astype(np.float64)
    # folds by scene (task, init): an explore episode and the pi0.5 rollouts of the same scene stay together
    sc_key = np.array([f"{m['tasks'][0]}_{m['ep'].split('_i')[1].split('_')[0]}" for m in M])
    uniq = np.unique(sc_key)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(uniq, rng.integers(0, args.folds, len(uniq))))
    fold = np.array([fold_of[s] for s in sc_key])
    ex = src == "explore"
    print(f"[data] {n} samples, explore {ex.sum()} (take {(ex & (out == 1)).sum()}, cont {(ex & (out == 0)).sum()}), "
          f"pi05 {(~ex).sum()}; explore success {y[ex].mean():.3f}", flush=True)
    res = {}
    for feat in ("low", "low+h"):
        for C in args.C:
            P = np.full((n, 2), np.nan)
            for f in range(args.folds):
                tr, te = fold != f, fold == f
                Z = low
                if feat == "low+h":
                    pca = PCA(16).fit(H[tr])
                    Z = np.c_[low, pca.transform(H) / 10.0]
                for o in (0, 1):
                    m_tr = tr & (out == o)
                    if o == 1:
                        m_tr = m_tr & ex
                    clf = LogisticRegression(C=C).fit(Z[m_tr], y[m_tr])
                    P[te, o] = clf.predict_proba(Z[te])[:, 1]
            r = {}
            for o, nm in ((0, "cont"), (1, "take")):
                m = ex & (out == o)
                r[f"auroc_{nm}_explore"] = round(T.auroc(y[m] > 0.5, P[m, o]), 3)
            m = (~ex)
            r["auroc_cont_pi05"] = round(T.auroc(y[m] > 0.5, P[m, 0]), 3)
            ips = {}
            idx = np.where(ex)[0]
            pr = np.where(out[idx] == 1, prop[idx], 1 - prop[idx])
            for mth in (-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2):
                want = (P[idx, 1] - P[idx, 0] > mth).astype(int)
                ips[f"m{mth:+.2f}"] = round(float(np.sum((want == out[idx]) * y[idx] / pr) / len(idx)), 3)
            r["ips"] = ips
            res[f"{feat}_C{C}"] = r
            print(f"[{feat} C={C}] " + json.dumps(r), flush=True)
    idx = np.where(ex)[0]
    pr = np.where(out[idx] == 1, prop[idx], 1 - prop[idx])
    base = {"always_take": float(np.sum((out[idx] == 1) * y[idx] / pr) / len(idx)),
            "never_take": float(np.sum((out[idx] == 0) * y[idx] / pr) / len(idx)),
            "logged_random": float(y[idx].mean()),
            "old_heads": float(np.sum(((X[idx, 0] - X[idx, 1] > 0).astype(int) == out[idx]) * y[idx] / pr) / len(idx))}
    # bootstrap the IPS standard error of always_take (by scene) as the noise scale
    sc = sc_key[idx]
    us = np.unique(sc)
    bs = []
    for _ in range(500):
        pick = rng.choice(us, len(us))
        sel = np.concatenate([np.where(sc == u)[0] for u in pick])
        bs.append(np.sum((out[idx][sel] == 1) * y[idx][sel] / pr[sel]) / len(sel))
    base["always_take_se"] = float(np.std(bs))
    print("[baselines] " + json.dumps({k: round(v, 3) for k, v in base.items()}), flush=True)
    json.dump(dict(res=res, base=base), open(f"{V}/value/cv_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
