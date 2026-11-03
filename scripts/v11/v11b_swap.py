"""
Does V11b's correction depend on this reference, or only on the current frame? (11-02, read only.) On validation camera and wide samples,
the reference images are replaced, gate held at its true value, everything else of the reference kept:
  moved    : the same reference, same step, with its camera moved (runs/v11/refvis): what a camera correction should react to;
  same_task: another reference of the same task, camera in place, its own aligned step; it carries the same view, so little should change
             if the branch reads the camera relation, and a change shows it reads this episode's layout;
  constant : one fixed image for every sample, the mean reference: the compare head then sees the current frame only;
  other    : another sample's reference (v11_paths.py), for comparison with the earlier probes.
    python v11b_swap.py runs/v11b/train/pma last.pt
"""
import argparse, collections, json, sys
import numpy as np
import torch
import train_v11b as B

run_dir, name = sys.argv[1:3]
args = argparse.Namespace(**json.load(open(f"{run_dir}/args.json")))
eps, refs, have = B.episodes(args)
va = {g: B.Mix(eps, refs, have, "val", args, args.val_n, group=g) for g in ("camera", "wide")}
ds = va["camera"]
run = B.Runner(args)
ck = run.load(f"{run_dir}/{name}")
run.modes(False)
pool = collections.defaultdict(list)
for k in sorted(have):
    pool[int(k.split("_i")[0][1:])].append(k)
keys = sorted(have)
m = []
for k in keys[:: max(len(keys) // 80, 1)]:
    with np.load(f"{args.refvis}/{k}_a0.npz") as z:
        m.append(z["vis"].astype(np.float32).mean(0))
mean = torch.from_numpy(np.mean(m, 0))
mean = mean / mean.norm(dim=-1, keepdim=True)
print("[swap] mean image from", len(m), "references", flush=True)


def moved(b):
    return torch.stack([torch.from_numpy(ds._small(f"{B.V11}/refvis/{k}_a0.npz")["vis"][int(j)].astype(np.float32)) for k, j in zip(b["rkey"], b["rrow"])])


def same_task(b):
    out = []
    for k, eid, row in zip(b["rkey"], b["eid"].tolist(), b["row"].tolist()):
        e = ds.eps[eid]
        p = [x for x in pool[e["task"]] if x != k]
        alt = p[int(np.random.default_rng([eid, row]).integers(len(p)))]
        rv = ds._small(f"{args.refvis}/{alt}_a0.npz")
        calls = np.where(rv["t"] % B.CALL == 0)[0]
        out.append(torch.from_numpy(rv["vis"][calls[ds.step_of(e, row, alt, rv["t"][calls])]].astype(np.float32)))
    return torch.stack(out)


D = collections.defaultdict(list)
with torch.no_grad():
    for g in va:
        for b in torch.utils.data.DataLoader(va[g], 16, shuffle=True, num_workers=2, drop_last=True, generator=torch.Generator().manual_seed(0)):
            tgt = b["target"].to(run.device)
            r = run.step(b)
            G = r["g"][:, None, None]
            D["full"].append(B.rms(G * r["delta"] - tgt).cpu()); D["zero"].append(B.rms(tgt).cpu()); D["kind"].append(b["kind"])
            D["no_compare"].append(B.rms(G * r["branch"] - tgt).cpu())
            for nm, v in (("moved", moved(b)), ("same_task", same_task(b)), ("constant", mean.expand(len(tgt), -1, -1)), ("other", torch.roll(b["ref_vis"].float(), 1, 0))):
                r2 = run.step(dict(b, ref_vis=v))
                D[nm].append(B.rms(G * r2["delta"] - tgt).cpu())
                D[nm + "|chg"].append((B.rms(r2["delta"] - r["delta"]) / (B.rms(r["delta"]) + 1e-9)).cpu())
                D[nm + "|chg_cmp"].append((B.rms(r2["compare"] - r["compare"]) / (B.rms(r["compare"]) + 1e-9)).cpu())
d = {k: torch.cat(v).float().numpy() for k, v in D.items()}
n = len(d["full"])
pick = np.random.default_rng(0).integers(0, n, (10000, n))
own = d["kind"] == B.KINDS.index("pi05_aug")
res = dict(run=run_dir, ckpt=name, step=ck.get("step"), n=n, zero=float(d["zero"].mean()), full=float(d["full"].mean()))
print(json.dumps(res), flush=True)
for k in ("moved", "same_task", "constant", "other", "no_compare"):
    e = d[k] - d["full"]
    lo, hi = np.percentile(e[pick].mean(1), [2.5, 97.5])
    res[k] = dict(err=float(d[k].mean()), rise=float(e.mean()), lo=float(lo), hi=float(hi), rise_pi05_own=float(e[own].mean()),
                  change=float(d[k + "|chg"].mean()) if k + "|chg" in d else None, change_compare=float(d[k + "|chg_cmp"].mean()) if k + "|chg_cmp" in d else None)
    print(k, json.dumps(res[k]), flush=True)
json.dump(res, open(f"{run_dir}/swap.json", "w"), indent=1)
