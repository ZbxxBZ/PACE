"""
V11 (10-27): what the residual a* - a_pi of the signal check (runs/v11/check/check.npz) is made of. CPU only, existing
data. All distances in pi0.5's normalized action space, on the executed part (first 10 steps of the chunk).
  0. by dimension: position (0-2), rotation (3-5), gripper (6), for a* - a_pi and for two samples of pi0.5 (spread).
  1. split a* - a_pi = (a* - a_ref) + (a_ref - a_pi), a_ref = the pair's reference at the same step (expert pairs
     replay the reference's timeline, so the robot at t is where the reference was at t, shifted by d).
  2. timing: a_ref read k steps later / earlier than t (k = -30 .. 30) vs a_pi; if some k != 0 is much closer, pi0.5
     plans the same motion at another pace / stage.
  3. pi0.5's own variety: two pi0.5 successes of the same task (different init states) at the same step, and at the best
     shift -- how far apart two correct ways of pi0.5 itself are.
The normalization (mean, std per dimension) is recovered from the stored normalized targets and the raw pair actions.
"""
import collections
import json

import numpy as np

V = "/root/pace/runs/v10b"
C = np.load("/root/pace/runs/v11/check/check.npz", allow_pickle=True)
EXEC, CHUNK = 10, 50
GROUPS = (("all", slice(0, 7)), ("pos", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7)))
SHIFTS = list(range(-30, 31, 5))


def chunk(a, t, n=CHUNK):
    t = max(int(t), 0)
    return np.stack([a[t + k] if t + k < len(a) else np.r_[np.zeros(6), a[-1][6]] for k in range(n)])


def rms(x, sl=slice(0, 7)):
    return float(np.sqrt((x[:EXEC, sl] ** 2).mean()))


_cache = {}


def load(path, field=None):
    if path not in _cache:
        if field:
            with np.load(path) as z:
                _cache[path] = np.array(z[field], np.float64)
        else:
            _cache[path] = np.load(path).astype(np.float64)
    return _cache[path]


kind, keys, ts = C["kind"], C["key"], C["t"].astype(int)
ex = np.where(kind == "expert")[0]
a_pi, a_pi2, tgt = (C[k].astype(np.float64) for k in ("a_pi", "a_pi2", "target"))
refs = {f"t{r['task_id']}_i{r['init']}": r for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))}

# normalization: target = (a* - mean) / std, fitted per dimension
X = np.concatenate([chunk(load(f"{V}/pairs/expert/{keys[i]}.npz", "actions"), ts[i]) for i in ex[:300]])
Y = np.concatenate([tgt[i] for i in ex[:300]])
fit = [np.polyfit(X[:, d], Y[:, d], 1) for d in range(7)]
std = np.array([1 / f[0] for f in fit])
mean = np.array([-f[1] / f[0] for f in fit])
norm = lambda a: (a - mean) / std
print("recovered mean", np.round(mean, 4), "std", np.round(std, 4))

out = collections.defaultdict(list)
best_k = []
for i in ex:
    rkey = keys[i].split("__")[1]
    ra = norm(load(f"{V}/refs/{refs[rkey]['actions']}"))
    t = ts[i]
    ref0 = chunk(ra, t)
    for g, sl in GROUPS:
        out[f"res_{g}"].append(rms(tgt[i] - a_pi[i], sl))
        out[f"spread_{g}"].append(rms(a_pi2[i] - a_pi[i], sl))
        out[f"dcorr_{g}"].append(rms(tgt[i] - ref0, sl))          # a* - a_ref: the d correction (+ feedback)
        out[f"refpi_{g}"].append(rms(ref0 - a_pi[i], sl))         # a_ref - a_pi
    ds = [rms(chunk(ra, t + k) - a_pi[i]) for k in SHIFTS]
    dp = [rms(chunk(ra, t + k) - a_pi[i], slice(0, 3)) for k in SHIFTS]
    out["shift_best_all"].append(min(ds))
    out["shift_best_pos"].append(min(dp))
    best_k.append(SHIFTS[int(np.argmin(ds))])
    sign_flip = np.mean((tgt[i][:EXEC, 6] > 0) != (a_pi[i][:EXEC, 6] > 0))
    out["grip_flip_frac"].append(float(sign_flip))
m = {k: np.mean(v) for k, v in out.items()}

lines = [f"expert samples: {len(ex)} (first {EXEC} steps, normalized)", "",
         "0. by dimension            all     pos     rot    grip"]
for nm, lab in (("res", "a* - a_pi (residual)"), ("spread", "two pi0.5 samples"), ("dcorr", "a* - a_ref (d corr.)"),
                ("refpi", "a_ref - a_pi")):
    lines.append(f"   {lab:22s} " + "  ".join(f"{m[f'{nm}_{g}']:.3f}" for g, _ in GROUPS))
lines.append(f"   gripper open/close disagreement a* vs a_pi: {100 * m['grip_flip_frac']:.1f}% of the executed steps")
lines += ["", "1. split: |a* - a_pi| = %.3f; |a* - a_ref| = %.3f (d correction); |a_ref - a_pi| = %.3f"
          % (m["res_all"], m["dcorr_all"], m["refpi_all"])]
c = collections.Counter(best_k)
lines += ["", "2. timing: a_ref shifted by k steps vs a_pi",
          f"   k = 0: {m['refpi_all']:.3f} (pos {m['refpi_pos']:.3f});  best k per sample: {m['shift_best_all']:.3f} "
          f"(pos {m['shift_best_pos']:.3f})",
          "   best k distribution: " + ", ".join(f"{k:+d}: {100 * c[k] / len(best_k):.0f}%" for k in SHIFTS)]

# 3. pi0.5's own variety between two of its successes of the same task
by_task = collections.defaultdict(list)
for k, r in refs.items():
    if r["success"]:
        by_task[r["task_id"]].append(k)
rng = np.random.default_rng(0)
var = collections.defaultdict(list)
for task, ks in sorted(by_task.items()):
    for _ in range(30):
        a, b = rng.choice(len(ks), 2, replace=False)
        A, B = norm(load(f"{V}/refs/{refs[ks[a]]['actions']}")), norm(load(f"{V}/refs/{refs[ks[b]]['actions']}"))
        for t in range(0, min(len(A), len(B)) - EXEC, 10):
            ca = chunk(A, t)
            for g, sl in GROUPS:
                var[g].append(rms(ca - chunk(B, t), sl))
            var["best"].append(min(rms(ca - chunk(B, t + k)) for k in SHIFTS))
lines += ["", "3. pi0.5's own variety: two pi0.5 successes of the same task (different init states), same step",
          "   " + "  ".join(f"{g} {np.mean(var[g]):.3f}" for g, _ in GROUPS) + f";  at the best shift {np.mean(var['best']):.3f}"
          + f"  ({len(var['all'])} windows)"]
txt = "\n".join(lines)
open("/root/pace/runs/v11/check/residual_parts.txt", "w").write(txt + "\n")
print(txt)
