"""
V10 premise 1 (10-21): do the evaluation's camera-viewpoint variants share their physical setup (robot start and object
layout) with the other variants of the same base task, so that a success recorded on another variant tells the right
robot-frame actions? CPU only, from the recorded states of V7's closed-loop runs on the same 140 variants
(plus_l10_V7_5500_acc3 and _other_acc3: 840 episodes; PACE barely changed the base's behaviour there).
  (a) robot start: the state at call 0 (after the env's settle steps) per variant vs the base task's background-texture
      variants (default robot, base init state);
  (b) object layout, through where successful episodes grasp: the end-effector position at the first call where the
      finger opening has shrunk by > 1 cm; a camera variant's success vs the nearest grasp of the same base task's
      same-layout categories (background, light, noise, language) and vs the nearest of its objects-layout variants;
      also where the camera variants' FAILED episodes first close the gripper.
    python v10_premise1.py
10-22 fix: the recorded states are the policy's normalized ones (MEAN_STD, pi0.5 preprocessor stats); they are mapped
back to the env's units (metres, radians, finger positions) before any distance, and a grasp is where the finger
opening has shrunk by > 1 cm (the 10-21 run measured normalized units and a 0.2 mm threshold).
"""
import collections
import glob
import json
import sys

import numpy as np
import torch

snap = glob.glob("/root/pace/cache/hf/hub/models--lerobot--pi05_libero_finetuned/snapshots/*/")[0]
from safetensors.torch import load_file  # noqa: E402

stats = load_file(glob.glob(snap + "policy_preprocessor*normalizer*.safetensors")[0])
MEAN, STD = stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()
OUT = sys.argv[1] if len(sys.argv) > 1 else "/root/pace/runs/rectest/v10_premise1.json"
RUNS = ["/root/pace/runs/eval/plus_l10_V7_5500_acc3", "/root/pace/runs/eval/plus_l10_V7_5500_other_acc3"]
SAME = ("Background Textures", "Light Conditions", "Sensor Noise", "Language Instructions")
eps = []
for run in RUNS:
    for f in sorted(glob.glob(f"{run}/rollouts/rollouts_*.pt")):
        for e in torch.load(f, map_location="cpu", weights_only=False)["episodes"]:
            m = e["meta"]
            eps.append(dict(task_id=int(m["task_id"]), cat=m["category"], base=m["base"], round=int(m["round"]),
                            success=bool(e["success"]), state=e["state"].float().numpy() * STD + MEAN, run=run.rsplit("_", 1)[-1]))
print(f"{len(eps)} episodes, {len({e['task_id'] for e in eps})} variants, {len({e['base'] for e in eps})} base tasks")


def grasp(state):
    w = state[:, 6] - state[:, 7]  # finger opening in metres (robosuite Panda: left > 0, right < 0)
    closed = np.nonzero(w < w[0] - 0.01)[0]
    return state[closed[0], :3] if len(closed) else None


# (a) robot start
ref = {}
for e in eps:
    if e["cat"] == "Background Textures":
        ref.setdefault(e["base"], []).append(e["state"][0])
ref = {b: np.median(np.stack(v), 0) for b, v in ref.items()}
print("\n(a) call-0 robot state vs the base task's background variants (max |diff| over the 8 dims, per episode)")
by = collections.defaultdict(list)
for e in eps:
    if e["base"] in ref:
        by[e["cat"]].append((float(np.abs(e["state"][0] - ref[e["base"]]).max()),
                             float(np.linalg.norm(e["state"][0, :3] - ref[e["base"]][:3]))))
for c, v in sorted(by.items()):
    v, pos = np.array([x for x, _ in v]), np.array([x for _, x in v])
    print(f"  {c:22s} n={len(v):3d}  median {np.median(v):.2e}  max {v.max():.2e}  share < 1e-3: {np.mean(v < 1e-3):.2f}"
          f"  [position: median {np.median(pos):.3f} m, min {pos.min():.3f} m]")

# (b) layout through grasp positions
g = collections.defaultdict(list)  # (base, group) -> grasp positions of successes
for e in eps:
    if e["success"]:
        p = grasp(e["state"])
        if p is not None:
            grp = "same" if e["cat"] in SAME else e["cat"]
            g[(e["base"], grp)].append(p)
near = lambda p, P: float(np.min(np.linalg.norm(np.stack(P) - p, axis=1))) if P else np.nan  # noqa: E731
rows = collections.defaultdict(list)
for e in eps:
    if e["cat"] != "Camera Viewpoints":
        continue
    p = grasp(e["state"])
    if p is None:
        rows["camera, no grasp"].append(np.nan)
        continue
    key = "camera success" if e["success"] else "camera failure"
    rows[key + " -> same-layout successes"].append(near(p, g[(e["base"], "same")]))
    rows[key + " -> objects-layout successes"].append(near(p, g[(e["base"], "Objects Layout")]))
# reference spreads: same-layout successes against each other (leave-one-out nearest)
loo = []
for (b, grp), P in g.items():
    if grp == "same" and len(P) > 1:
        A = np.stack(P)
        D = np.linalg.norm(A[:, None] - A[None], axis=-1) + np.eye(len(A)) * 1e9
        loo += D.min(1).tolist()
rows["same-layout success -> other same-layout successes"] = loo
ol = []
for e in eps:
    if e["cat"] == "Objects Layout" and e["success"]:
        p = grasp(e["state"])
        if p is not None:
            ol.append(near(p, g[(e["base"], "same")]))
rows["objects-layout success -> same-layout successes"] = ol
print("\n(b) nearest grasp position (end effector, metres): median / 75% / n")
for k, v in rows.items():
    v = np.array([x for x in v if x == x])
    if len(v):
        print(f"  {k:52s} {np.median(v):.3f} / {np.percentile(v, 75):.3f} / {len(v)}")
    else:
        print(f"  {k:52s} n=0")

# coverage: for every camera variant, same-layout successes of its base task in the runs (what a store would hold)
print("\n(c) per camera variant: base task, outcomes of its episodes, same-layout successes of the base task")
cam = sorted({(e["task_id"], e["base"]) for e in eps if e["cat"] == "Camera Viewpoints"})
for t, b in cam:
    outs = [e["success"] for e in eps if e["task_id"] == t]
    print(f"  {t:5d} {b[:46]:46s} success {sum(outs)}/{len(outs)}; same-layout successes {len(g[(b, 'same')])}")
json.dump({k: [float(x) for x in v] for k, v in rows.items()}, open(OUT, "w"))
