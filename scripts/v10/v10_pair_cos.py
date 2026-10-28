"""V10-0 check (10-22, CPU): call-0 image-token cosine (agentview / wrist) of pairs the follower would match (same
instruction told, same call-0 robot state), split into truly same physical setup (non-objects-layout pairs, premise 1-2)
and different layout (objects-layout pairs); V7's recorded episodes."""
import collections
import glob

import numpy as np
import torch

RUNS = ["/root/pace/runs/eval/plus_l10_V7_5500_acc3", "/root/pace/runs/eval/plus_l10_V7_5500_other_acc3"]
first = {}
for run in RUNS:
    for f in sorted(glob.glob(f"{run}/rollouts/rollouts_*.pt")):
        for e in torch.load(f, map_location="cpu", weights_only=False)["episodes"]:
            t = int(e["meta"]["task_id"])
            if t not in first:
                first[t] = dict(cat=e["meta"]["category"], key=e["task_key"], s0=e["state"][0].float(), vis=e["vis"][0].float())


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
        kind = "layout pair (objects layout)" if a["cat"] == "Objects Layout" else \
            "same setup: " + " / ".join(sorted({a["cat"].split()[0], b["cat"].split()[0]}))
        pairs[kind].append(cos_cams(a["vis"], b["vis"]))
print(f"{'pair kind':40s} {'n':>4s}  agentview min/median/max      wrist min/median/max")
for k in sorted(pairs):
    v = np.array(pairs[k])
    print(f"{k:40s} {len(v):4d}  {v[:,0].min():.4f} {np.median(v[:,0]):.4f} {v[:,0].max():.4f}    "
          f"{v[:,1].min():.4f} {np.median(v[:,1]):.4f} {v[:,1].max():.4f}")
