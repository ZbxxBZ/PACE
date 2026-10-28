import collections
import json

import numpy as np

T = json.load(open("/root/pace/runs/rectest/v10_shift/noncam_targets.json"))
cat = {int(t["task_id"]): t["category"] for t in T}
r = [json.loads(l) for l in open("/root/pace/runs/v10b/closed/noncam/episodes.jsonl")]
cell = collections.defaultdict(lambda: collections.defaultdict(list))
for x in r:
    cell[(x["task_id"], x["init"])][x["mode"]].append(x)
APPEAR = ("Light Conditions", "Background Textures", "Sensor Noise", "Language Instructions")


def line(name, cs):
    out = [f"{name:22s} cells {len(cs):3d}"]
    b = np.array([np.mean([e["success"] for e in cell[c]["base"]]) for c in cs])
    out.append(f"base {100 * b.mean():5.1f}")
    for mode in ("v10", "v10v"):
        v = np.array([np.mean([e["success"] for e in cell[c][mode]]) for c in cs])
        var = np.array([c[0] for c in cs])
        uv = np.unique(var)
        rng = np.random.default_rng(0)
        bs = [(v - b)[np.concatenate([np.where(var == u)[0] for u in rng.choice(uv, len(uv))])].mean() for _ in range(2000)]
        eps = [e for c in cs for e in cell[c][mode]]
        fol = sum(bool(e.get("follow")) for e in eps)
        dm = [e for c in cs if sorted(cell[c]["base"], key=lambda x: x["seed"])[0]["success"] for e in cell[c][mode]]
        out.append(f"{mode} {100 * v.mean():5.1f} ({100 * (v - b).mean():+.1f} [{100 * np.percentile(bs, 2.5):+.1f},"
                   f"{100 * np.percentile(bs, 97.5):+.1f}], follow {fol}/{len(eps)}, damage "
                   f"{100 * np.mean([not e['success'] for e in dm]):.1f}%)")
    print("  ".join(out))


done = [c for c in cell if cell[c]["v10v"] and cell[c]["v10"] and len(cell[c]["base"]) == 2]
for c_ in sorted({cat[c[0]] for c in done}):
    line(c_, [c for c in done if cat[c[0]] == c_])
line("appearance (4 cats)", [c for c in done if cat[c[0]] in APPEAR])
line("all done cells", done)
