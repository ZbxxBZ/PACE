"""
Gate 1 of docs/PACE_V10_经验配准方案.md (10-23): the true-d upper bound on LIBERO-PRO's position swap, entirely in
LIBERO-PRO's own code base (Python 3.8), experience recorded there too (pro_eval.py --save_actions on the original
libero_10: pi0.5's successes, env-unit actions). For every libero_10_swap task x init state (seed 1000 + 100*task +
init, the same scenes as the base run), up to --per_target experiences of the same task are tried:
  replay : the experience's actions, open loop;
  track0 : follow the experience's hand path (KP * error / 0.05 m added to its actions), no shift;
  trackd : the same path shifted per phase by the true displacement of that phase's goal anchor between the
           experience's scene and the target scene (simulator positions after the settle steps; privileged).
Phases come from the experience's gripper command changes; anchors from the task's BDDL goal (the corrected rule of
v10_oracle_shift2.py, unchanged): grasp -> the object grasped; release -> the carried object's goal destination (a fixed
table region gives d = 0; letting go and re-grasping near the pick spot keeps the object); last phase -> the fixture of
a "close" goal, else the last destination. Every experience is first replayed in its own scene and must succeed.
EGL on the GPU (small cameras: images are not used).
    python pro_oracle.py --workers 16 -> runs/rectest/v10_pro/oracle_l10_swap.json
"""
import argparse
import collections
import json
import multiprocessing as mp
import random
import time

import numpy as np

import pro_common as C

EXP = "/root/pace/runs/rectest/v10_pro/exp_l10"
OUT = "/root/pace/runs/rectest/v10_pro/oracle_l10_swap.json"
SIZE = 64


class Sim:
    def __init__(self, env):
        self.inner = env.env
        self.robot = self.inner.robots[0]
        self.sim = self.inner.sim
        self.movable = list(self.inner.objects_dict.keys())
        self.bodies = dict(self.inner.obj_body_id)

    def hand(self):
        return np.array(self.sim.data.site_xpos[self.robot.eef_site_id], dtype=np.float64)

    def positions(self):
        return {n: np.array(self.sim.data.body_xpos[i], dtype=np.float64) for n, i in self.bodies.items()}


def task_index(suite_name):
    s = C.suite_of(suite_name)
    return {s.get_task(i).name: i for i in range(s.n_tasks)}


def record_source(src):
    env, task, inits = C.make_env("libero_10", src["task"], SIZE)
    try:
        C.reset(env, inits[src["init"]], src["seed"])
        sim = Sim(env)
        acts = src["actions"]
        start = sim.positions()
        path, events, success, end = [], [], False, len(acts)
        closed = acts[0][6] > 0
        for t, a in enumerate(acts):
            path.append(sim.hand())
            now = a[6] > 0
            if now != closed and all((x[6] > 0) == now for x in acts[t:t + 3]):
                events.append(dict(t=t, kind="grasp" if now else "release", hand=sim.hand(), pos=sim.positions()))
                closed = now
            env.step(a)
            if env.check_success():
                success, end = True, t + 1
                break
        final = dict(t=end, hand=sim.hand(), pos=sim.positions())
        return dict(success=success, end=end, path=np.stack(path), events=events, final=final, start=start,
                    movable=sim.movable, goal=[list(g) for g in env.env.parsed_problem["goal_state"]])
    finally:
        env.close()


def anchor_body(arg, bodies):
    cand = [b for b in bodies if arg == b or arg.startswith(b + "_")]
    return max(cand, key=len) if cand else arg


def phase_targets(rec, regrasp_dist=0.05):
    bodies, evs = list(rec["start"]), rec["events"]
    dest = {g[1]: anchor_body(g[2], bodies) for g in rec["goal"] if len(g) == 3}
    closes = [anchor_body(g[1], bodies) for g in rec["goal"] if len(g) == 2 and g[0].lower() == "close"]
    near = lambda ev, cand: min(cand, key=lambda b: np.linalg.norm(ev["pos"][b][:2] - ev["hand"][:2]))
    grasped = [near(ev, rec["movable"]) if ev["kind"] == "grasp" else None for ev in evs]
    out, carried, t0, last_dest = [], None, 0, None
    for i, ev in enumerate(evs):
        if ev["kind"] == "grasp":
            name = carried = grasped[i]
        else:
            nxt = next((grasped[j] for j in range(i + 1, len(evs)) if evs[j]["kind"] == "grasp"), None)
            if (carried is not None and nxt == carried
                    and np.linalg.norm(ev["hand"][:2] - rec["start"][carried][:2]) < regrasp_dist):
                name = carried
            elif carried in dest:
                name = last_dest = dest[carried]
            else:
                name = last_dest = near(ev, [b for b in ev["pos"] if b != carried])
            carried = None
        out.append((t0, ev["t"], name))
        t0 = ev["t"]
    f = rec["final"]
    if closes:
        name = closes[0]
    elif carried is not None:
        name = dest.get(carried, carried)
    else:
        name = last_dest or min(f["pos"], key=lambda b: np.linalg.norm(f["pos"][b][:2] - f["hand"][:2]))
    out.append((t0, f["t"], name))
    return out


def shift_vector(rec, phases, target_start, n, blend):
    d_phase = []
    for _, _, b in phases:
        d = np.zeros(3)
        if b in target_start and b in rec["start"]:
            d[:2] = target_start[b][:2] - rec["start"][b][:2]
        d_phase.append(d)
    dv = np.zeros((n, 3))
    prev = d_phase[0]
    for (s, e, _), d in zip(phases, d_phase):
        for t in range(s, min(max(e, s + 1), n)):
            w = min(1.0, (t - s) / max(blend, 1)) if s > 0 else 1.0
            dv[t] = (1 - w) * prev + w * d
        prev = d
    last = phases[-1][1]
    if last < n:
        dv[last:] = d_phase[-1]
    return dv, [(s, e, b, [round(float(x), 4) for x in d[:2]]) for (s, e, b), d in zip(phases, d_phase)]


def run_episode(task_idx, init_idx, seed, actions, mode, path=None, dvec=None, kp=1.0, hold=40):
    env, _, inits = C.make_env("libero_10_swap", task_idx, SIZE)
    try:
        C.reset(env, inits[init_idx], seed)
        sim = Sim(env)
        n = len(actions)
        for t in range(n + hold):
            if t < n:
                a = np.array(actions[t], dtype=np.float64)
            else:
                a = np.zeros(7)
                a[6] = actions[-1][6]
            if mode == "track":
                tgt = path[min(t, n - 1)] + dvec[min(t, n - 1)]
                a[:3] = np.clip(a[:3] + kp * (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            env.step(a)
            if env.check_success():
                return True, t + 1
        return False, n + hold
    finally:
        env.close()


def target_start(task_idx, init_idx, seed):
    env, _, inits = C.make_env("libero_10_swap", task_idx, SIZE)
    try:
        C.reset(env, inits[init_idx], seed)
        return Sim(env).positions()
    finally:
        env.close()


def job(args):
    (tid, k), sources, kp, blend = args
    seed = 1000 + 100 * tid + k
    tstart = target_start(tid, k, seed)
    out = []
    for src in sources:
        rec, acts = src["rec"], src["actions"]
        n = len(acts)
        phases = phase_targets(rec)
        dvec, dinfo = shift_vector(rec, phases, tstart, n, blend)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        row = dict(target=tid, init=k, source=src["key"], phases=dinfo)
        ok, steps = run_episode(tid, k, seed, acts, "replay")
        row["replay"] = dict(success=ok, steps=steps)
        ok, steps = run_episode(tid, k, seed, acts, "track", path, np.zeros((n, 3)), kp)
        row["track0"] = dict(success=ok, steps=steps)
        ok, steps = run_episode(tid, k, seed, acts, "track", path, dvec, kp)
        row["trackd"] = dict(success=ok, steps=steps)
        out.append(row)
    return out


def rec_job(src):
    return src["key"], record_source(src)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--per_target", type=int, default=3)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--blend", type=int, default=15)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{EXP}/episodes.jsonl")]
    arrays = np.load(f"{EXP}/actions.npz")
    orig = task_index("libero_10")
    swap = task_index("libero_10_swap")
    # experience: up to per_target + 2 successes per original task (spares for a failed self-replay), by init state
    srcs = []
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        key = f"t{r['task']}_i{r['init']}"
        if r["success"] and key in arrays.files and sum(s["task"] == r["task"] for s in srcs) < args.per_target + 2:
            srcs.append(dict(key=key, task=r["task"], name=r["name"], init=r["init"], seed=r["seed"],
                             actions=arrays[key]))
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        recs = dict(pool.map(rec_job, srcs))
        for s in srcs:
            s["rec"] = recs[s["key"]]
        good = [s for s in srcs if s["rec"]["success"]]
        print(f"[experience] {len(srcs)} replayed in their own scene, {len(good)} succeed again "
              f"({time.time() - t0:.0f}s)", flush=True)
        jobs = []
        for name, tid in sorted(swap.items(), key=lambda x: x[1]):
            pool_src = [s for s in good if s["name"] == name][:args.per_target]
            for k in args.inits:
                jobs.append(((tid, k), pool_src, args.kp, args.blend))
        print(f"[jobs] {len(jobs)} (task, init) x up to {args.per_target} experiences; tasks without experience: "
              f"{[n for n in swap if not any(s['name'] == n for s in good)]}", flush=True)
        results = []
        for out in pool.imap_unordered(job, jobs):
            results += out
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"[progress] {len(results)} pairs ({time.time() - t0:.0f}s): " + ", ".join(
                f"{m} {sum(r[m]['success'] for r in results)}" for m in ("replay", "track0", "trackd")), flush=True)
    summarize(results)


def summarize(results):
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    names = {i: n for n, i in task_index("libero_10_swap").items()}
    cells = collections.defaultdict(list)
    for r in results:
        cells[(r["target"], r["init"])].append(r)
    mean = lambda ks, m: float(np.mean([np.mean([r[m]["success"] for r in cells[k]]) if m != "base" else base[k]
                                        for k in ks]))
    keys = sorted(cells)
    summary = {"cells": len(keys), "pairs": len(results)}
    for m in ("base", "replay", "track0", "trackd"):
        summary[m] = round(100 * mean(keys, m), 1)
    tasks = sorted({k[0] for k in keys})
    rng = random.Random(0)
    for a, b in (("trackd", "base"), ("trackd", "replay")):
        diff = lambda ts: mean([k for t in ts for k in keys if k[0] == t], a) - mean([k for t in ts for k in keys if k[0] == t], b)
        bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
        summary[f"{a}-{b}"] = [round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)]
    summary["per_task"] = {f"{t} {names[t][:60]}": {m: round(100 * mean([k for k in keys if k[0] == t], m))
                                                    for m in ("base", "replay", "track0", "trackd")} for t in tasks}
    json.dump(summary, open(OUT.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
