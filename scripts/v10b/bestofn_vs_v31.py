import collections
import json

import numpy as np

C = "/root/pace/runs/v10b/closed"
B = [json.loads(l) for l in open(f"{C}/bestofn/episodes.jsonl")]
bc = collections.defaultdict(dict)
for x in B:
    bc[(x["task_id"], x["init"])][(x["seed"] - 1000 - 7 * x["task_id"] - x["init"]) // 4000] = x["success"]
V = collections.defaultdict(list)
for x in map(json.loads, open(f"{C}/soft/episodes.jsonl")):
    if x["mode"] == "v10v":
        V[(x["task_id"], x["init"])].append(x["success"])
cs = sorted(V)
var = np.array([c[0] for c in cs])
b = np.array([np.mean([bc[c][0], bc[c][1]]) for c in cs])
v = np.array([np.mean(V[c]) for c in cs])
p5 = np.array([max(bc[c].values()) for c in cs])
uv = np.unique(var)
rng = np.random.default_rng(0)


def ci(d):
    bs = [d[np.concatenate([np.where(var == u)[0] for u in rng.choice(uv, len(uv))])].mean() for _ in range(4000)]
    return f"{100 * d.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]"


print(f"V10-3-1 {100 * v.mean():.1f}  base {100 * b.mean():.1f}  pass@5 {100 * p5.mean():.1f}")
print("V10-3-1 - base:", ci(v - b), "  V10-3-1 - pass@5:", ci(v - p5))
allfail = [i for i, c in enumerate(cs) if max(bc[c].values()) == 0]
print(f"cells where pi0.5 fails all 5 seeds: {len(allfail)}; V10-3-1 success there {100 * v[allfail].mean():.1f}% "
      f"(cells with >= 1 success: {sum(v[i] > 0 for i in allfail)})")
allok = [i for i, c in enumerate(cs) if min(bc[c].values()) == 1]
print(f"cells where pi0.5 succeeds all 5: {len(allok)}; V10-3-1 there {100 * v[allok].mean():.1f}%")
