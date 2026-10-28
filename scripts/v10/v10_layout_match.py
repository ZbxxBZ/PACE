"""
V10-0 check (10-22, CPU): which variants share the instruction told and the call-0 robot state (what the follower
treats as the same physical setup), from V7's recorded closed-loop episodes on the evaluation's 140 variants; and
whether such a pair really shows the same scene: cosine of their call-0 image tokens (v2 encoder input: 16 cells per
camera, agentview then wrist), per camera, against the spread between rounds of one variant (same scene).
    python v10_layout_match.py
"""
import collections
import glob

import numpy as np
import torch
from safetensors.torch import load_file

RUNS = ["/root/pace/runs/eval/plus_l10_V7_5500_acc3", "/root/pace/runs/eval/plus_l10_V7_5500_other_acc3"]
snap = glob.glob("/root/pace/cache/hf/hub/models--lerobot--pi05_libero_finetuned/snapshots/*/")[0]
stats = load_file(glob.glob(snap + "policy_preprocessor*normalizer*.safetensors")[0])
MEAN, STD = stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()
first = {}  # task_id -> call-0 record (first episode seen)
for run in RUNS:
    for f in sorted(glob.glob(f"{run}/rollouts/rollouts_*.pt")):
        for e in torch.load(f, map_location="cpu", weights_only=False)["episodes"]:
            t = int(e["meta"]["task_id"])
            rec = dict(name=e["meta"]["name"], cat=e["meta"]["category"], key=e["task_key"],
                       s0=e["state"][0].float().numpy() * STD + MEAN,
                       vis=e["vis"][0].float() if "vis" in e else None)
            first.setdefault(t, []).append(rec)


def cos_cams(a, b):
    a, b = a.view(2, 16, -1), b.view(2, 16, -1)
    return [float(torch.nn.functional.cosine_similarity(a[c], b[c], dim=-1).mean()) for c in range(2)]


same_scene = [cos_cams(r[0]["vis"], r[1]["vis"]) for r in first.values() if len(r) > 1 and r[0]["vis"] is not None]
print(f"same variant, two episodes (same scene): call-0 token cosine agentview median {np.median([x[0] for x in same_scene]):.4f} "
      f"min {min(x[0] for x in same_scene):.4f}; wrist median {np.median([x[1] for x in same_scene]):.4f} "
      f"min {min(x[1] for x in same_scene):.4f}")
ids = sorted(first)
groups = collections.defaultdict(list)
for t in ids:
    r = first[t][0]
    groups[r["key"]].append(t)
print("\nper objects-layout variant: other variants with the same instruction and call-0 state (max |diff| < 1e-3)")
for t in ids:
    r = first[t][0]
    if r["cat"] != "Objects Layout":
        continue
    match = [u for u in groups[r["key"]] if u != t and np.abs(first[u][0]["s0"] - r["s0"]).max() < 1e-3]
    desc = []
    for u in match:
        o = first[u][0]
        c = cos_cams(r["vis"], o["vis"]) if r["vis"] is not None and o["vis"] is not None else [float("nan")] * 2
        desc.append(f"{u}({o['cat'][:7]},{o['name'][-28:]}; cos agent {c[0]:.4f} wrist {c[1]:.4f})")
    print(f"  {t:5d} {r['name'][-34:]:34s} -> {len(match)}: " + "; ".join(desc))
