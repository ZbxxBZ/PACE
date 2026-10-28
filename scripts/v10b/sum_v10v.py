import collections
import json
import sys

import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "/root/pace/runs/v10b/closed/run2"
r = [json.loads(l) for l in open(f"{R}/episodes.jsonl")]
q = [x for x in r if x["mode"] == "v10v"]
nfs, dec = 0, collections.Counter()
for x in q:
    z = json.load(open(f"{R}/wrapper/{x['key']}.json"))
    nfs += bool(z["from_start"])
    for c in z["calls"]:
        if "decision" in c:
            dec[(c["decision"], c["go"])] += 1
print(f"v10v: {len(q)} episodes, followed from start {nfs}, decisions {dict(dec)}")
fs = [x for x in q if json.load(open(f"{R}/wrapper/{x['key']}.json"))["from_start"]]
print(f"  followed from start: {len(fs)} episodes, success {sum(x['success'] for x in fs)}; not: "
      f"{len(q) - len(fs)}, success {sum(x['success'] for x in q) - sum(x['success'] for x in fs)}")
pv = {(x["task_id"], x["init"], x["source"]): x["success"] for x in r if x["mode"] == "v10"}
pairs = collections.Counter((x["success"], pv.get((x["task_id"], x["init"], x["source"]))) for x in q)
print("  paired with V10-1 (v10v, v10):", dict(pairs))
# per base task (variant -> base via the targets file)
tg = {json.loads(l)["task_id"]: json.loads(l)["name"].split("_view_")[0] for l in open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl")}
by = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    if x["mode"] in ("base", "v10", "v10v") and x["task_id"] in tg:
        by[tg[x["task_id"]]][x["mode"]].append(x["success"])
for b, v in sorted(by.items()):
    print(f"  {b[:60]:60s} base {100 * np.mean(v['base']):5.1f}  V10-1 {100 * np.mean(v['v10']):5.1f}  V10-3 {100 * np.mean(v['v10v']):5.1f}")
# V10-3 minus V10-1, clustered by variant
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    cell[(x["task_id"], x["init"])][x["mode"]].append(x["success"])
cs = [c for c in cell if cell[c]["v10"] and cell[c]["v10v"]]
dlt = np.array([np.mean(cell[c]["v10v"]) - np.mean(cell[c]["v10"]) for c in cs])
var = np.array([c[0] for c in cs])
uv = np.unique(var)
rng = np.random.default_rng(0)
bs = [dlt[np.concatenate([np.where(var == u)[0] for u in rng.choice(uv, len(uv))])].mean() for _ in range(4000)]
print(f"  V10-3 - V10-1: {100 * dlt.mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")
