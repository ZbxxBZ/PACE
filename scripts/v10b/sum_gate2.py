"""Gate 2 (original libero_10, closed_loop_v3.py --orig): V10-3 vs pi0.5 on the same GPU, per cell (task, init)."""
import collections
import json
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/root/pace/runs/v10b/closed/gate2"
r = [json.loads(l) for l in open(f"{R}/episodes.jsonl")]
print("GPU:", dict(collections.Counter(x.get("gpu") for x in r)))
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    cell[(x["task_id"], x["init"])][x["mode"]].append(x)
cs = sorted(c for c in cell if cell[c]["v10v"] and cell[c]["base"])
b = np.array([np.mean([e["success"] for e in cell[c]["base"]]) for c in cs])
v = np.array([np.mean([e["success"] for e in cell[c]["v10v"]]) for c in cs])
task = np.array([c[0] for c in cs])
ut = np.unique(task)
rng = np.random.default_rng(0)
bs = [(v - b)[np.concatenate([np.where(task == u)[0] for u in rng.choice(ut, len(ut))])].mean() for _ in range(4000)]
eps = [e for c in cs for e in cell[c]["v10v"]]
fol = sum(bool(e.get("follow")) for e in eps)
dm = [e for c in cs if sorted(cell[c]["base"], key=lambda x: x["seed"])[0]["success"] for e in cell[c]["v10v"]]
print(f"cells {len(cs)}  base {100 * b.mean():.1f}  V10-3 {100 * v.mean():.1f}  diff {100 * (v - b).mean():+.1f} "
      f"[{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}] (task-clustered)  "
      f"follow {fol}/{len(eps)}  damage {100 * np.mean([not e['success'] for e in dm]):.1f}% ({len(dm)} eps)")
print("gate 2 (V10-3 >= base - 2 pp):", "PASS" if (v - b).mean() >= -0.02 else "FAIL")
for t in ut:
    m = task == t
    fe = [e for c in cs if c[0] == t for e in cell[c]["v10v"]]
    print(f"  task {t}: base {100 * b[m].mean():5.1f}  V10-3 {100 * v[m].mean():5.1f}  "
          f"follow {sum(bool(e.get('follow')) for e in fe)}/{len(fe)}")
