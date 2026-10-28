"""
Section 7.2 of docs/PACE_V10_经验配准方案.md (registered 10-23 16:45): "bring the hand there, let pi0.5 do the
contact" -- an upper bound with simulator truth on libero_10_swap (10 tasks x init states 0..9, the base run's seeds),
one experience per task (its first success on the original libero_10). The experience gives the phase order (gripper
events, anchors by the corrected rule of pro_oracle.py) and, for every grasp / release, where the hand was PRE steps
before it relative to the phase's anchor. Per phase:
  1. stage: rise to the experience's highest hand height, move above "experience hand PRE steps before the contact +
     the anchor's true displacement (live simulator position now vs. the experience at that contact)", descend there
     (proportional control, no rotation, gripper held);
  2. contact, two ways:
     vla : pi0.5 takes over until it commands this phase's gripper change (3 steps in a row); the last phase (e.g.
           closing a door) runs pi0.5 until success or the budget;
     exp : follow the experience's path from PRE steps before the contact to the contact (+5 steps), shifted by the
           same displacement (control on control).
After every contact the hand holds still for GRIP_SETTLE steps with the new gripper command, so the gripper finishes
closing / opening before the next staging lifts away (the smoke run left objects behind without it).
Budget 520 steps in total (staging included); at most 150 pi0.5 steps per phase.
    python pro_stage.py --workers 10 -> runs/rectest/v10_pro/stage_l10_swap.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import random
import time

import numpy as np

import pro_common as C
import pro_oracle as P

OUT = "/root/pace/runs/rectest/v10_pro/stage_l10_swap.json"
PRE = 20
LEG_STEPS = 60
VLA_PHASE_STEPS = 150
GRIP_SETTLE = 15
BUDGET = 520


class Episode:
    def __init__(self, task_idx, init_idx, seed, client):
        self.env, self.task, inits = C.make_env("libero_10_swap", task_idx, P.SIZE if client is None else C.SIZE)
        self.obs = C.reset(self.env, inits[init_idx], seed)
        self.sim = P.Sim(self.env)
        self.seed, self.client, self.calls = seed, client, 0
        self.t, self.success, self.grip = 0, False, -1.0

    def step(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.obs, _, _, _ = self.env.step(a)
        self.t += 1
        self.grip = float(np.sign(a[6])) or self.grip
        if self.env.check_success():
            self.success = True
        return self.success or self.t >= BUDGET

    def goto(self, target):
        for _ in range(LEG_STEPS):
            err = target - self.sim.hand()
            if np.linalg.norm(err) < 0.01:
                return False
            a = np.zeros(7)
            a[:3] = np.clip(err / 0.05, -1.0, 1.0)
            a[6] = self.grip
            if self.step(a):
                return True
        return False

    def stage(self, target, z_safe):
        h = self.sim.hand()
        for tgt in (np.array([h[0], h[1], max(h[2], z_safe)]), np.array([target[0], target[1], max(h[2], z_safe)]),
                    target):
            if self.goto(tgt):
                return True
        return False

    def vla_until(self, kind):
        """pi0.5 acts until it commands the phase's gripper change for 3 steps in a row (kind None: never)."""
        want = 1.0 if kind == "grasp" else -1.0
        run, chunk = 0, None
        for i in range(VLA_PHASE_STEPS if kind else BUDGET):
            if i % 10 == 0:
                chunk = self.client.act(self.env, self.obs, self.task.language, self.seed, self.calls)
                self.calls += 1
            a = chunk[i % 10]
            if self.step(a):
                return True
            run = run + 1 if (kind and np.sign(a[6]) == want) else 0
            if kind and run >= 3:
                return False
        return False

    def close(self):
        self.env.close()


def plan(rec):
    """Per phase: (kind, contact step, pre-contact step, anchor, anchor position in the experience at the contact)."""
    phases = P.phase_targets(rec)
    out = []
    for i, (s, e, anchor) in enumerate(phases):
        if i < len(rec["events"]):
            ev = rec["events"][i]
            kind, t_ev, pos = ev["kind"], ev["t"], ev["pos"]
        else:
            kind, t_ev, pos = None, rec["final"]["t"], rec["final"]["pos"]
        out.append(dict(kind=kind, t=t_ev, pre=max(s, t_ev - PRE), anchor=anchor, anchor_pos=pos.get(anchor)))
    return out


def run_cell(job):
    (t, k), src, mode, port = job
    seed = 1000 + 100 * t + k
    client = C.PolicyClient(port) if mode == "vla" else None
    ep = Episode(t, k, seed, client)
    rec, acts = src["rec"], src["actions"]
    path = rec["path"]
    z_safe = float(path[:, 2].max())
    reached = 0
    try:
        for ph in plan(rec):
            d = np.zeros(3)
            if ph["anchor_pos"] is not None:
                d[:2] = ep.sim.positions()[ph["anchor"]][:2] - ph["anchor_pos"][:2]
            pre_hand = path[min(ph["pre"], len(path) - 1)]
            if ep.stage(pre_hand + d, z_safe):
                break
            if mode == "vla":
                done = ep.vla_until(ph["kind"])
            else:
                done = False
                end = min(len(acts), ph["t"] + 5) if ph["kind"] else len(acts)
                for j in range(ph["pre"], end):
                    a = np.array(acts[j], dtype=np.float64)
                    a[:3] = np.clip(a[:3] + (path[min(j, len(path) - 1)] + d - ep.sim.hand()) / 0.05, -1.0, 1.0)
                    if ep.step(a):
                        done = True
                        break
            reached += 1
            if done:
                break
            for _ in range(GRIP_SETTLE):  # let the gripper finish closing / opening before lifting away
                a = np.zeros(7)
                a[6] = ep.grip
                if ep.step(a):
                    done = True
                    break
            if done:
                break
        if not ep.success and mode == "vla" and ep.t < BUDGET:  # all phases staged: let pi0.5 finish the task
            ep.vla_until(None)
        return dict(task=t, init=k, mode=mode, source=src["key"], success=ep.success, steps=ep.t,
                    phases=len(plan(rec)), reached=reached)
    finally:
        ep.close()
        if client:
            client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--modes", nargs="+", default=["vla", "exp"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    swap = P.task_index("libero_10_swap")
    first = {}
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        if r["success"] and r["name"] not in first:
            first[r["name"]] = dict(key=f"t{r['task']}_i{r['init']}", task=r["task"], init=r["init"], seed=r["seed"],
                                    actions=arrays[f"t{r['task']}_i{r['init']}"])
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, src in first.items():
            src["rec"] = P.record_source(src)
            assert src["rec"]["success"], name
        print(f"[experience] {len(first)} tasks, one experience each ({time.time() - t0:.0f}s)", flush=True)
        jobs = [((tid, k), first[name], m, args.port) for m in args.modes for name, tid in sorted(swap.items(), key=lambda x: x[1])
                for k in args.inits]
        results = []
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"[progress] {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s): " + ", ".join(
                f"{m} {sum(x['success'] for x in results if x['mode'] == m)}/{sum(x['mode'] == m for x in results)}"
                for m in args.modes), flush=True)
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    names = {i: n for n, i in swap.items()}
    summary = {"base": round(100 * np.mean(list(base.values())), 1)}
    rng = random.Random(0)
    tasks = sorted(names)
    for m in args.modes:
        sel = {(r["task"], r["init"]): r["success"] for r in results if r["mode"] == m}
        summary[m] = round(100 * np.mean(list(sel.values())), 1)
        diff = lambda ts: np.mean([sel[c] - base[c] for c in sel if c[0] in ts])
        bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
        summary[f"{m}-base"] = [round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)]
    summary["per_task"] = {f"{t} {names[t][:55]}": {"base": round(100 * np.mean([base[(t, k)] for k in args.inits]))}
                           for t in tasks}
    for m in args.modes:
        for t in tasks:
            summary["per_task"][f"{t} {names[t][:55]}"][m] = round(100 * np.mean(
                [r["success"] for r in results if r["mode"] == m and r["task"] == t]))
    json.dump(summary, open(OUT.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
