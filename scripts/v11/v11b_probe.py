"""V11b design check (10-31, read only): on validation samples of run2's best.pt, is "this reference matches" readable from
the mean of h_on - h_off (where a match head would sit); how large is Delta per reference mode; how far is the arm's progress
from the time-aligned reference call on the training-style pairs (vs 10-60 steps median in the closed-loop trace)."""
import argparse, collections, json
import numpy as np
import torch
import train_v11_v9 as M

T = M.T
args = argparse.Namespace(**json.load(open("runs/v11/train/v9_run2_4090/args.json")))
kinds = list(args.kinds)
args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
idx = T.build_index(args)
args.kinds = kinds
idx = M.add_pi05_aug(idx)
N, D = 300, [0, 1, 2, 6, 7]
va = {m: M.V11Data(idx, "val", args, False, mode=m, limit=N) for m in ("true", "other", "unaligned")}
run = M.Runner(args)
ck = torch.load("runs/v11/train/v9_run2_4090/best.pt", map_location="cpu", weights_only=False)
run.adapter.load_state_dict(ck["adapter"]); run.head.load_state_dict(ck["head"]); run.adapter.eval()
cap = {}
def pre(mod, inp):
    a, b = inp
    cap["f"] = ((a - b) / (b.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()).mean(1).float().cpu()
run.head.register_forward_pre_hook(pre)
orig = run.batch_state
def bs(o):
    st = orig(o); cap["st"] = st.detach().float().cpu(); return st
run.batch_state = bs
out, off, rel = {}, collections.defaultdict(list), {}
with torch.no_grad():
    for m, ds in va.items():
        F, K, R, E, n = [], [], [], [], 0
        st = []
        for b in torch.utils.data.DataLoader(ds, 8, shuffle=False, num_workers=2):
            delta, _, stats = run(b)
            F.append(cap["f"]); K.append(b["kind"]); R.append((delta[:, :10] ** 2).mean((1, 2)).sqrt().cpu())
            st.append(stats.get("relevance_mean", float("nan")))
            for i in range(len(b["t"])):
                s = idx["samples"][ds.items[n]]; ep = idx["eps"][s["ep"]]; E.append(s["ep"])
                if m == "true" and ep["kind"] != "negative":
                    rv = ds._small(f"{args.refvis}/{ep['ref']}_a0.npz")
                    tr = np.asarray(rv["t"]); c = tr % 10 == 0
                    dist = np.linalg.norm(np.asarray(rv["state"])[c][:, D] - cap["st"][i, D].numpy(), axis=1)
                    tc = tr[c]; off[ep["kind"]].append(abs(int(tc[np.argmin(dist)]) - int(tc[np.argmin(np.abs(tc - s["t"]))])))
                n += 1
        out[m] = dict(f=torch.cat(F).numpy(), k=torch.cat(K).numpy(), r=torch.cat(R).numpy(), e=np.array(E))
        rel[m] = float(np.nanmean(st))
        print(f"[{m}] n {n} Delta rms (executed) overall {out[m]['r'].mean():.3f}; by kind", {M.KINDS[k]: round(float(out[m]['r'][out[m]['k'] == k].mean()), 3) for k in sorted(set(out[m]['k'].tolist()))}, "relevance", round(rel[m], 3), flush=True)
print("progress offset on training-style pairs (steps): median p90 by kind", {k: (float(np.median(v)), float(np.percentile(v, 90)), len(v)) for k, v in off.items()}, flush=True)
def auc(s, y):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    p, q = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - p * (p + 1) / 2) / (p * q))
keep = lambda d: d["k"] != M.KINDS.index("negative")
for neg in ("other", "unaligned"):
    A, B = out["true"], out[neg]
    X = np.r_[A["f"][keep(A)], B["f"][keep(B)]]; y = np.r_[np.ones(keep(A).sum()), np.zeros(keep(B).sum())]; e = np.r_[A["e"][keep(A)], B["e"][keep(B)]]
    res = []
    for fold in (0, 1):
        tr, te = e % 2 == fold, e % 2 != fold
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        U, S, Vt = np.linalg.svd((X[tr] - mu) / sd, full_matrices=False)
        Z = ((X - mu) / sd) @ Vt[:32].T
        Z = Z / (Z[tr].std(0) + 1e-6)
        w = torch.zeros(33, requires_grad=True, dtype=torch.float64)
        Zt = torch.tensor(np.c_[Z[tr], np.ones(tr.sum())]); yt = torch.tensor(y[tr])
        opt = torch.optim.LBFGS([w], max_iter=200)
        def cl():
            opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(Zt @ w, yt) + 1e-2 * w[:-1].pow(2).sum(); l.backward(); return l
        opt.step(cl)
        res.append(auc((np.c_[Z[te], np.ones(te.sum())] @ w.detach().numpy()), y[te]))
    print(f"AUC true vs {neg} from mean(h_on-h_off), held-out episodes: {res[0]:.3f} {res[1]:.3f}; from Delta size alone {auc(np.r_[A['r'][keep(A)], B['r'][keep(B)]], y):.3f}", flush=True)
