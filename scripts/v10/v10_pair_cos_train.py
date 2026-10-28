"""V10-0 gate calibration (10-22, CPU) on the V7 training collection (runs/rollouts_v7/*, variants disjoint from the
evaluation's 140): pairs of variants the follower would match (same instruction told, same call-0 robot state), split
into same physical setup (not objects layout) and different layout (objects-layout pairs); their call-0 image-token
cosine per camera; how a wrist-cosine threshold separates them. Files are memory-mapped (only call 0 is read)."""
import collections
import glob

import numpy as np
import torch

first = {}
for f in sorted(glob.glob("/root/pace/runs/rollouts_v7/*/rollouts/rollouts_*.pt")):
    if "/smoke/" in f:
        continue
    for e in torch.load(f, map_location="cpu", weights_only=False, mmap=True)["episodes"]:
        t = int(e["meta"]["task_id"])
        if t not in first and "vis" in e:
            first[t] = dict(cat=e["meta"]["category"], key=e["task_key"], s0=e["state"][0].float().clone(),
                            vis=e["vis"][0].float().clone())
print(f"{len(first)} training variants; categories {collections.Counter(v['cat'] for v in first.values())}")


def cos_cams(a, b):
    a, b = a.view(2, 16, -1), b.view(2, 16, -1)
    return [float(torch.nn.functional.cosine_similarity(a[c], b[c], dim=-1).mean()) for c in range(2)]


pairs = collections.defaultdict(list)
ids = sorted(first)
for i, t in enumerate(ids):
    for u in ids[i + 1:]:
        a, b = first[t], first[u]
        if a["key"] != b["key"] or float((a["s0"] - b["s0"]).abs().max()) > 1e-3:
            continue
        if a["cat"] == "Objects Layout" or b["cat"] == "Objects Layout":
            kind = "layout pair" if a["cat"] == b["cat"] else "objects layout / other (?)"
        else:
            kind = "same setup: " + " / ".join(sorted({a["cat"].split()[0], b["cat"].split()[0]}))
        pairs[kind].append(cos_cams(a["vis"], b["vis"]))
print(f"{'pair kind':40s} {'n':>5s}  agentview min/median/max      wrist min/median/max")
for k in sorted(pairs):
    v = np.array(pairs[k])
    print(f"{k:40s} {len(v):5d}  {v[:,0].min():.4f} {np.median(v[:,0]):.4f} {v[:,0].max():.4f}    "
          f"{v[:,1].min():.4f} {np.median(v[:,1]):.4f} {v[:,1].max():.4f}")
lay = np.array(pairs.get("layout pair", [[0, 0]]))[:, 1]
print("\nwrist threshold: share of layout pairs passing / share of same-setup pairs passing (camera pairs only)")
for th in (0.995, 0.996, 0.997, 0.998, 0.9985, 0.999):
    same = np.concatenate([np.array(v)[:, 1] for k, v in pairs.items() if k.startswith("same setup")])
    cam = np.concatenate([np.array(v)[:, 1] for k, v in pairs.items() if k.startswith("same setup") and "Camera" in k])
    print(f"  {th:.4f}: layout {np.mean(lay >= th):.3f}   same setup {np.mean(same >= th):.3f}   camera pairs {np.mean(cam >= th):.3f}")
