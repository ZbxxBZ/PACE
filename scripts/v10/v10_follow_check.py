"""
V10-0 follower, offline check (10-22, CPU, no env): pace/follow.py's reference pick and alignment on V7's recorded
closed-loop episodes (plus_l10_V7_5500_acc3 and _other_acc3, the evaluation's 140 variants).
  (1) self: every success against a store holding only itself -> exact match, followed on every call at distance 0
      with its own chunks;
  (2) reach: per category, a reference among the OTHER variants' successes (what a store holds in the run) passes
      the gates (instruction, call-0 robot state, wrist image), and of which category it is;
  (3) failure memory: after a failed follow of the picked reference in a scene, the next pick there avoids it;
  (4) camera episodes' own recorded states against the picked reference: where the base's own trajectory parts from
      it (the base did not follow: this only shows the alignment code at work, not what following would do).
    python v10_follow_check.py
"""
import collections
import glob

import numpy as np
import torch
from safetensors.torch import load_file

from pace.follow import Follower, _Row

RUNS = ["/root/pace/runs/eval/plus_l10_V7_5500_acc3", "/root/pace/runs/eval/plus_l10_V7_5500_other_acc3"]
snap = glob.glob("/root/pace/cache/hf/hub/models--lerobot--pi05_libero_finetuned/snapshots/*/")[0]
stats = load_file(glob.glob(snap + "policy_preprocessor*normalizer*.safetensors")[0])

eps = []
for run in RUNS:
    for f in sorted(glob.glob(f"{run}/rollouts/rollouts_*.pt")):
        eps += torch.load(f, map_location="cpu", weights_only=False)["episodes"]
print(f"{len(eps)} episodes; task keys e.g. {eps[0]['task_key'][:60]!r}; actions {tuple(eps[0]['actions'].shape)}")
cat_of = {int(e["meta"]["task_id"]): e["meta"]["category"] for e in eps}


class FakeStore:
    def __init__(self, episodes):
        self.episodes = episodes


class FakePace:
    store = None


def follower():
    f = Follower.__new__(Follower)
    f.mean, f.std = stats["observation.state.mean"].numpy(), stats["observation.state.std"].numpy()
    f.max_dist, f.window, f.setup_tol, f.wrist_cos, f.exact_cos = 0.02, 3, 1e-3, 0.999, 0.99999
    f.pace, f.rows, f.runtimes, f.vis0, f.failed = FakePace(), None, None, {}, []
    return f


def run(f, ep):
    """Feed the episode's recorded (normalized) states call by call; the chunks the follower would return."""
    row = _Row(ep["task_key"])
    out = [f.step(row, f.env_units(ep["state"][c]), ep["vis"][0] if c == 0 else None)
           for c in range(ep["state"].shape[0])]
    return row, out


succ = [e for e in eps if e["success"]]
# (1) self
f, bad = follower(), 0
for e in succ:
    f.pace.store = FakeStore([e])
    row, out = run(f, e)
    T = e["state"].shape[0]
    bad += not (row.ref is e and row.kind == "exact" and row.followed == T and row.status == "follow"
                and max(row.dists) == 0.0 and all(torch.equal(o, e["actions"][c]) for c, o in enumerate(out)))
print(f"(1) self: {len(succ) - bad}/{len(succ)} successes picked as exact matches and followed on every call at "
      f"distance 0 with their own chunks")

# (2) reach per category (first episode of every variant), store = the other variants' successes
f = follower()
reach, ref_cats = collections.Counter(), collections.defaultdict(collections.Counter)
seen = set()
for e in eps:
    t = int(e["meta"]["task_id"])
    if t in seen:
        continue
    seen.add(t)
    f.pace.store = FakeStore([o for o in succ if int(o["meta"]["task_id"]) != t])
    (ref, kind, wrist), n = f.pick(e["task_key"], f.env_units(e["state"][0]), e["vis"][0])
    reach[(e["meta"]["category"], ref is not None)] += 1
    if ref is not None:
        ref_cats[e["meta"]["category"]][cat_of[int(ref["meta"]["task_id"])]] += 1
print("(2) other-variant reference passing the gates (per variant):")
for cat in sorted({c for c, _ in reach}):
    print(f"  {cat:22s} yes {reach[(cat, True)]:3d}  no {reach[(cat, False)]:3d}  reference categories "
          f"{dict(ref_cats[cat])}")

# (3) failure memory
f = follower()
e = next(o for o in eps if o["meta"]["category"] == "Camera Viewpoints")
f.pace.store = FakeStore([o for o in succ if int(o["meta"]["task_id"]) != int(e["meta"]["task_id"])])
(r1, _, _), _ = f.pick(e["task_key"], f.env_units(e["state"][0]), e["vis"][0])
f.failed.append((e["vis"][0], r1["name"]))
(r2, _, _), _ = f.pick(e["task_key"], f.env_units(e["state"][0]), e["vis"][0])
print(f"(3) failure memory: first pick {r1['name'][-40:]}, after its failure {r2['name'][-40:] if r2 else None} "
      f"({'avoided' if r2 is None or r2['name'] != r1['name'] else 'NOT avoided'})")

# (4) camera episodes vs the picked reference
f = follower()
status, frac = collections.Counter(), []
for e in eps:
    if e["meta"]["category"] != "Camera Viewpoints":
        continue
    f.pace.store = FakeStore([o for o in succ if int(o["meta"]["task_id"]) != int(e["meta"]["task_id"])])
    row, _ = run(f, e)
    status[row.status] += 1
    if row.ref is not None:
        frac.append(row.followed / e["state"].shape[0])
print(f"(4) camera episodes' own trajectories vs the picked reference: {dict(status)}; share of calls within 2 cm "
      f"median {np.median(frac):.2f}")
