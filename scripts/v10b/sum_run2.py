import collections
import json

import numpy as np

R = "/root/pace/runs/v10b/closed/run2"
r = [json.loads(l) for l in open(f"{R}/episodes.jsonl")]
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    cell[(x["task_id"], x["init"])][x["mode"]].append(x)
for mode in ("v10", "v10p", "v10q", "v10v"):
    cs = [c for c, v in cell.items() if v[mode] and v["base"]]
    if not cs:
        continue
    dm = [e for c in cs if sorted(cell[c]["base"], key=lambda x: x["seed"])[0]["success"] for e in cell[c][mode]]
    v = np.array([np.mean([e["success"] for e in cell[c][mode]]) for c in cs])
    b = np.array([np.mean([e["success"] for e in cell[c]["base"]]) for c in cs])
    var = np.array([c[0] for c in cs])
    uv = np.unique(var)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(4000):
        s = rng.choice(uv, len(uv))
        idx = np.concatenate([np.where(var == u)[0] for u in s])
        bs.append((v[idx] - b[idx]).mean())
    print(f"{mode}: cells {len(cs)} success {100 * v.mean():.1f} vs base {100 * b.mean():.1f}, diff {100 * (v - b).mean():+.1f} "
          f"[{100 * np.percentile(bs, 2.5):+.1f},{100 * np.percentile(bs, 97.5):+.1f}], damage "
          f"{100 * np.mean([not e['success'] for e in dm]):.1f}% of {len(dm)}")
# V10-2b decisions
q = [x for x in r if x["mode"] == "v10q"]
if q:
    nfs, nb, ncont, nhb, hb_succ = 0, 0, 0, 0, []
    for x in q:
        z = json.load(open(f"{R}/wrapper/{x['key']}.json"))
        nfs += bool(z["from_start"])
        b = [c for c in z["calls"] if "boundary" in c]
        nb += len(b)
        ncont += sum(bool(c["go"]) for c in b)
        if any(not c["go"] for c in b):
            nhb += 1
            hb_succ.append(x["success"])
    print(f"v10q: {len(q)} episodes, from_start {nfs}, boundary decisions {nb} (continue {ncont}), "
          f"episodes with a hand-back {nhb} (success {sum(hb_succ)})")
    pv = {(x["task_id"], x["init"], x["source"]): x["success"] for x in r if x["mode"] == "v10"}
    pairs = collections.Counter((x["success"], pv.get((x["task_id"], x["init"], x["source"]))) for x in q)
    print("paired with V10-1 (v10q, v10):", dict(pairs))
