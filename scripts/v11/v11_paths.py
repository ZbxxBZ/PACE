"""
V11 branch check (11-01, read only): on validation samples of a run, what happens to Delta when a path of the branch is
switched off (input injection, cross-attention) or one input of the reference is taken from another sample (action chunk,
image tokens, state + position). Batches are shuffled, so a swap mostly brings another task's reference. Errors are RMS over the
executed 10 steps; the differences to the full branch get a bootstrap interval over samples (positive samples).
    python v11_paths.py runs/v11/train/b_iso last.pt runs/v11/probe/b_iso.json
"""
import argparse, collections, json, os, sys
import numpy as np
import torch
import train_v11_v9 as M

run_dir, name, out_json = sys.argv[1:4]
T = M.T
args = argparse.Namespace(**json.load(open(f"{run_dir}/args.json")))
kinds = list(args.kinds)
args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
idx = T.build_index(args)
args.kinds = kinds
idx = M.add_pi05_aug(idx)
ds = M.V11Data(idx, "val", args, False, mode="true", limit=320)
run = M.Runner(args)
ck = torch.load(f"{run_dir}/{name}", map_location="cpu", weights_only=False)
run.adapter.load_state_dict(ck["adapter"]); run.head.load_state_dict(ck["head"]); run.adapter.eval()
A = run.adapter
inj = A.inject_proj.weight.data.clone()
xat = {k: l.o_proj.weight.data.clone() for k, l in A.layers.items()}
def paths(i=True, x=True):
    A.inject_proj.weight.data.copy_(inj if i else torch.zeros_like(inj))
    for k, l in A.layers.items():
        l.o_proj.weight.data.copy_(xat[k] if x else torch.zeros_like(xat[k]))
SW = dict(act=["a_ref"], vis=["ref_vis"], state=["ref_state", "ref_pos"], all=["a_ref", "ref_vis", "ref_state", "ref_pos"])
V = collections.OrderedDict(full=(True, True, None), no_inject=(False, True, None), no_xattn=(True, False, None),
                            swap_act=(True, True, "act"), swap_vis=(True, True, "vis"), swap_state=(True, True, "state"), swap_all=(True, True, "all"))
out, gate = collections.defaultdict(list), collections.defaultdict(list)
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for b in torch.utils.data.DataLoader(ds, 8, shuffle=True, generator=g, num_workers=2, drop_last=True):
        tgt = b["target"][:, :10].float()
        D = {}
        for nm, (i, x, sw) in V.items():
            paths(i, x)
            c = dict(b)
            if sw:
                for k in SW[sw]:
                    c[k] = torch.roll(b[k], 1, 0)
            r = run(c)
            D[nm] = r[0][:, :10].float().cpu()
            gate[nm].append(r[2].get("relevance_mean", float("nan")))
        paths()
        a = D["full"]
        cp = b["a_ref"][:, :10].float() - b["a_pi"][:, :10].float()
        n = lambda x: x.flatten(1).norm(dim=1)
        for nm, d in D.items():
            out[nm + "|rms"].append((d ** 2).mean((1, 2)).sqrt())
            out[nm + "|err"].append(((d - tgt) ** 2).mean((1, 2)).sqrt())
            out[nm + "|cos"].append((d * a).flatten(1).sum(1) / (n(d) * n(a) + 1e-9))
            out[nm + "|chg"].append(n(d - a) / (n(a) + 1e-9))
            out[nm + "|copy"].append((d * cp).flatten(1).sum(1) / (n(d) * n(cp) + 1e-9))
        out["kind"].append(b["kind"])
        out["zero"].append((tgt ** 2).mean((1, 2)).sqrt())
K = torch.cat(out["kind"]).numpy()
neg = K == M.KINDS.index("negative")
arr = lambda k: torch.cat(out[k]).numpy()
f = lambda k, s: float(arr(k)[s].mean())
res = dict(run=run_dir, ckpt=name, step=ck.get("step"), samples=len(K), negative=int(neg.sum()), zero=f("zero", ~neg), variants={})
print(f"{run_dir}/{name} step {ck.get('step')}, samples {len(K)}, negative {neg.sum()}; error of Delta=0: positive {res['zero']:.4f}", flush=True)
print("variant     rms(pos)  err(pos)  err(neg)  cos_to_full  change/|full|  cos_to_(a_ref-a_pi)  gate   err-full [95%]", flush=True)
rng = np.random.default_rng(0)
base = arr("full|err")[~neg]
for nm in V:
    e = arr(nm + "|err")[~neg] - base
    s = e[rng.integers(0, len(e), (10000, len(e)))].mean(1)
    lo, hi = np.percentile(s, [2.5, 97.5])
    res["variants"][nm] = dict(rms=f(nm + "|rms", ~neg), err=f(nm + "|err", ~neg), err_neg=f(nm + "|err", neg), cos=f(nm + "|cos", ~neg), change=f(nm + "|chg", ~neg),
                               copy=f(nm + "|copy", ~neg), gate=float(np.nanmean(gate[nm])), diff=float(e.mean()), lo=float(lo), hi=float(hi))
    v = res["variants"][nm]
    print(f" {nm:10s} {v['rms']:.4f}   {v['err']:.4f}   {v['err_neg']:.4f}   {v['cos']:+.3f}       {v['change']:.3f}        {v['copy']:+.3f}       {v['gate']:.3f}  {v['diff']:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
os.makedirs(os.path.dirname(out_json), exist_ok=True)
json.dump(res, open(out_json, "w"), indent=1)
