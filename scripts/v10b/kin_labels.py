"""
d labels a real robot can produce (10-24, second version): instead of the simulator's object displacement (BDDL
anchors), the difference of the HAND position at matching grasp / release events of two successful pi0.5 episodes --
the current scene's own pi0.5 success (runs/v10b/refs has one per init state) and the reference. The hand position
comes from the arm's kinematics, so no object has to be measured.
  * events must match: same sequence of grasp / release, and the same object grasped at each grasp (nearest movable
    body at the grasp); otherwise the pair gets no label (masked in training);
  * phase p of the reference (ending at its event p) gets hand_cur(event p) - hand_ref(event p) (xy); the last phase
    (no gripper event at its end) gets the final-hand difference when the task has a "close" goal (drawer /
    microwave), else the previous phase's label (the last release's destination);
  * per step, phases blend over 15 steps exactly like v10_oracle_shift2.shift_vector.
Writes runs/v10b/kin_labels/<key>.npy (n, 2) for every expert / perturbed pair whose events match, and prints how the
kinematic labels compare with the true (BDDL-anchor) d per phase kind.
    python kin_labels.py
"""
import collections
import json
import os
import pickle

import numpy as np

V = "/root/pace/runs/v10b"


def grasped(rec, ev):
    return min(rec["movable"], key=lambda b: np.linalg.norm(ev["pos"][b][:2] - ev["hand"][:2]))


def signature(rec):
    return [(ev["kind"], grasped(rec, ev) if ev["kind"] == "grasp" else None) for ev in rec["events"]]


def blend(phases, vecs, n, blend_steps=15):
    dv = np.zeros((n, 2))
    prev = vecs[0]
    for (s, e), d in zip(phases, vecs):
        for t in range(s, min(max(e, s + 1), n)):
            w = min(1.0, (t - s) / max(blend_steps, 1)) if s > 0 else 1.0
            dv[t] = (1 - w) * prev + w * d
        prev = d
    if phases[-1][1] < n:
        dv[phases[-1][1]:] = vecs[-1]
    return dv


def main():
    recs = pickle.load(open(f"{V}/pairs/refs_rec.pkl", "rb"))
    os.makedirs(f"{V}/kin_labels", exist_ok=True)
    stats, kinds = collections.Counter(), collections.defaultdict(list)
    for kind in ("expert", "perturbed"):
        for r in map(json.loads, open(f"{V}/pairs/{kind}.jsonl")):
            stats[kind, "pairs"] += 1
            cur, ref = recs.get((r["task_id"], r["init"])), recs.get((r["ref_task"], r["ref_init"]))
            if cur is None or not cur["success"]:
                stats[kind, "no pi0.5 success in the current scene"] += 1
                continue
            if signature(cur) != signature(ref):
                stats[kind, "events differ"] += 1
                continue
            z = np.load(f"{V}/pairs/{kind}/{r['key']}.npz")
            phases, d_true, n = [tuple(p) for p in z["phases"]], z["d"][:, :2], len(z["d"])
            assert len(phases) == len(ref["events"]) + 1, (len(phases), len(ref["events"]))
            vecs = [cur["events"][i]["hand"][:2] - ref["events"][i]["hand"][:2] for i in range(len(ref["events"]))]
            close = any(len(g) == 2 and g[0].lower() == "close" for g in ref["goal"])
            vecs.append(cur["final"]["hand"][:2] - ref["final"]["hand"][:2] if close else vecs[-1])
            dk = blend(phases, vecs, n)
            np.save(f"{V}/kin_labels/{r['key']}.npy", dk.astype(np.float32))
            stats[kind, "labelled"] += 1
            if kind == "expert":
                for i, (s, e) in enumerate(phases):
                    t = min(max(e - 1, s), n - 1)
                    lab = "last" if i == len(phases) - 1 else ref["events"][i]["kind"]
                    kinds[lab].append(100 * np.linalg.norm(dk[t] - d_true[t]))
    for k, v in sorted(stats.items()):
        print(f"[{k[0]}] {k[1]}: {v}")
    print("\n[kinematic label - true d] at the end of each phase, expert pairs (cm): median / 75th / 90th, n")
    for lab, v in sorted(kinds.items()):
        v = np.array(v)
        print(f"  {lab:8s} {np.median(v):5.2f} / {np.percentile(v, 75):5.2f} / {np.percentile(v, 90):5.2f}  n={len(v)}")
    print(f"-> {V}/kin_labels")


if __name__ == "__main__":
    main()
