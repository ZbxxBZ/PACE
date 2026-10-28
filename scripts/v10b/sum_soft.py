"""best-of-N (runs/v10b/closed/bestofn) and the V10-3-2 soft-follow sweep (runs/v10b/closed/soft), RTX 4080 (10-27).
Registered in the desktop design doc 10.14b. Base = bestofn seeds 0, 1 (seed, seed + 4000) of every camera cell."""
import collections
import json
import os
import sys

import numpy as np

C = "/root/pace/runs/v10b/closed"
SOFT = sys.argv[1] if len(sys.argv) > 1 else f"{C}/soft"


def load(d):
    p = f"{d}/episodes.jsonl"
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def ci(diff, var, n=4000):
    uv = np.unique(var)
    rng = np.random.default_rng(0)
    bs = [diff[np.concatenate([np.where(var == u)[0] for u in rng.choice(uv, len(uv))])].mean() for _ in range(n)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


B = load(f"{C}/bestofn")
print("GPU:", dict(collections.Counter(x.get("gpu") for x in B)))
cell = collections.defaultdict(dict)
for x in B:
    s0 = 1000 + 7 * x["task_id"] + x["init"]
    cell[(x["task_id"], x["init"])][(x["seed"] - s0) // 4000] = x["success"]
full = sorted(c for c, v in cell.items() if len(v) == 5)
print(f"best-of-N: {len(full)} / {len(cell)} cells with 5 seeds")
if full:
    S = np.array([[cell[c][i] for i in range(5)] for c in full], dtype=float)
    for n in range(1, 6):
        print(f"  pass@{n}: {100 * S[:, :n].max(1).mean():5.1f}%   (mean success of seeds 0..{n - 1}: "
              f"{100 * S[:, :n].mean():5.1f}%)")
    print(f"  cells where all 5 fail: {100 * (S.max(1) == 0).mean():.1f}%; all 5 succeed: {100 * (S.min(1) == 1).mean():.1f}%")

R = load(SOFT)
if not R:
    sys.exit()
arm = collections.defaultdict(lambda: collections.defaultdict(list))
for x in R:
    arm[x["mode"]][(x["task_id"], x["init"])].append(x)
base = {c: np.mean([v[0], v[1]]) for c, v in cell.items() if 0 in v and 1 in v}
base_s0 = {c: v[0] for c, v in cell.items() if 0 in v}
modes = sorted(arm, key=lambda m: (m != "v10v", int(m[4:]) if m.startswith("v10s") else -1))
ref = arm.get("v10v", {})
cs = sorted(set(base).intersection(*[set(arm[m]) for m in modes]))
var = np.array([c[0] for c in cs])
b = np.array([base[c] for c in cs])
print(f"\nsoft sweep: {len(cs)} cells in all arms; base {100 * b.mean():.1f}%")
fs_keys = None
for m in modes:
    v = np.array([np.mean([e["success"] for e in arm[m][c]]) for c in cs])
    lo, hi = ci(v - b, var)
    line = f"  {m:7s} {100 * v.mean():5.1f}%  vs base {100 * (v - b).mean():+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    if m != "v10v" and ref:
        r_ = np.array([np.mean([e["success"] for e in ref[c]]) for c in cs])
        lo, hi = ci(v - r_, var)
        line += f"  vs V10-3-1 {100 * (v - r_).mean():+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    eps = [e for c in cs for e in arm[m][c]]
    dm = [e for c in cs if base_s0.get(c) for e in arm[m][c]]
    wr = {e["key"]: json.load(open(f"{SOFT}/wrapper/{e['key']}.json")) for e in eps
          if os.path.exists(f"{SOFT}/wrapper/{e['key']}.json")}
    fs = [e for e in eps if wr.get(e["key"], {}).get("from_start")]
    line += (f"  damage {100 * np.mean([not e['success'] for e in dm]):.1f}%  follow-from-start {len(fs)}/{len(eps)}"
             f" success {100 * np.mean([e['success'] for e in fs]) if fs else float('nan'):.1f}%")
    sd = [wr[k]["soft"] for k in wr if wr[k].get("soft") and wr[k]["soft"]["n"]]
    if sd:
        line += (f"  dev_pos {np.mean([s['dev_pos'] for s in sd]):.4f}"
                 f" grip_flip {100 * np.mean([s['grip_flip'] for s in sd]):.1f}%")
    print(line)
# the followed-from-start set, paired by (cell, source) across arms
if ref:
    fs_ref = {(e["task_id"], e["init"], e["source"]) for c in cs for e in ref[c]
              if json.load(open(f"{SOFT}/wrapper/{e['key']}.json")).get("from_start")}
    print(f"\n  episodes V10-3-1 follows from the start: {len(fs_ref)}; success on exactly these, per arm:")
    for m in modes:
        s_ = [e["success"] for c in cs for e in arm[m][c] if (e["task_id"], e["init"], e["source"]) in fs_ref]
        print(f"    {m:7s} {100 * np.mean(s_):5.1f}% ({len(s_)})")
