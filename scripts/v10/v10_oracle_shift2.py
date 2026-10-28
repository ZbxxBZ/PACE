"""
V10 true-d upper bound, corrected anchors (10-23; docs/PACE_V1-V8.md, "决定继续训练 V10-1b"). Same as
v10_oracle_shift.py (same targets, sources, seeds and conditions) except which body each phase follows, now taken from
the task's BDDL goal instead of "the body nearest the hand":
  grasp phase   : the object grasped (nearest movable, unchanged);
  release phase : the carried object's goal destination (basket / plate / cabinet / microwave / caddy); a destination
                  that is a fixed table region (e.g. living_room_table_plate_right_region) gives d = 0; letting go and
                  grasping the same object again near where it started keeps that object;
  last phase    : the fixture of a "close" goal if there is one, else the last released object's destination.
Writes runs/rectest/v10_shift/oracle_shift2.json. The original description follows.

V10 true-d upper bound (10-22; registered in docs/PACE_V1-V8.md, "下一步：跨扰动对照与泛化测试"): if the shift d of a
reference were known exactly, would following the shifted reference succeed where the objects have moved?
Targets: the evaluation's 20 camera-viewpoint variants, each from init states 1..5 (the evaluation uses 0; objects move
2-4 cm between init states, v10_initstate_spread.py). Sources: successes recorded by the V10-0 run
(plus_l10_V10f_acc3) on the same base task's light / sensor-noise / background variants, all from init state 0: the
same objects, in other places. Up to --per_target sources (distinct variants) per target. No policy anywhere:
  replay : the source's executed actions, open loop;
  track0 : follow the source's end-effector path, each step its action plus a proportional correction toward the
           path's point (KP * error / 0.05 m, the OSC controller's position scale), d = 0;
  trackd : the same, the path shifted per phase by the true displacement d of that phase's object between the two
           scenes (simulator positions after the settle steps: privileged, an upper bound only).
Phases: the source's gripper command changes (open -> close = grasp, close -> open = release). A phase ending in a grasp
aims at the object grasped (the movable object nearest the hand then); one ending in a release at the body nearest the
hand then (objects and fixtures, the carried object excluded); the last phase at the body nearest the hand at the end.
d blends into the next phase's over --blend steps. Every source is first replayed in its own env (must succeed: exact
replay) to record its per-step hand path, gripper events and the bodies' positions at the events.
    python v10_oracle_shift2.py --workers 8
"""
import argparse
import collections
import functools
import glob
import json
import multiprocessing as mp
import random
import time

import numpy as np
import torch

RUN = "/root/pace/runs/eval/plus_l10_V10f_acc3"
SOURCE_CATS = ("Light Conditions", "Sensor Noise", "Background Textures")
G = {}  # per worker: env factory pieces


def worker_init():
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.libero import _get_suite, _make_env_fns, parse_camera_names

    env_cfg = LiberoPlusEnv(task="libero_10")
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    G.update(env_cfg=env_cfg, suite=_get_suite("libero_10"), gk=gk, make_fns=_make_env_fns,
             cams=parse_camera_names(env_cfg.camera_name))


def make_env(tid, init_index):
    c = G["env_cfg"]
    fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=G["cams"],
                       episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=G["gk"],
                       control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                       is_libero_plus=c.is_libero_plus)[0]
    kw = dict(fn.keywords)
    kw.update(observation_width=64, observation_height=64)  # images are not used: keep rendering cheap
    return fn.func(int(init_index), **kw)


class Sim:
    """Handles into a reset LiberoEnv: hand position, gripper opening, body positions."""

    def __init__(self, env):
        self.inner = env._env.env
        self.robot = self.inner.robots[0]
        self.sim = self.inner.sim
        self.movable = list(self.inner.objects_dict.keys())
        self.bodies = dict(self.inner.obj_body_id)  # objects and fixtures

    def hand(self):
        return np.array(self.sim.data.site_xpos[self.robot.eef_site_id], dtype=np.float64)

    def positions(self):
        return {n: np.array(self.sim.data.body_xpos[i], dtype=np.float64) for n, i in self.bodies.items()}


def run_episode(tid, init_index, seed, actions, mode, path=None, dvec=None, kp=1.0, hold=40):
    """Execute in a fresh env of (tid, init_index). mode replay | track. Returns (success, steps)."""
    env = make_env(tid, init_index)
    try:
        env.reset(seed=int(seed))
        sim = Sim(env)
        n = len(actions)
        for t in range(n + hold):
            if t < n:
                a = actions[t].copy()
            else:
                a = np.zeros(7)
                a[6] = actions[-1][6]
            if mode == "track":
                tgt = path[min(t, n - 1)] + dvec[min(t, n - 1)]
                a[:3] = np.clip(a[:3] + kp * (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            _, _, _, _, info = env.step(a)
            if info.get("is_success"):
                return True, t + 1
        return False, n + hold
    finally:
        env.close()


def record_source(src):
    """Replay a source in its own env; per-step hand path, gripper events and body positions at them."""
    env = make_env(src["task_id"], src["init_index"])
    try:
        env.reset(seed=int(src["seed"]))
        sim = Sim(env)
        acts = src["actions"]
        start = sim.positions()
        path, events, success, end = [], [], False, len(acts)
        closed = acts[0][6] > 0
        for t, a in enumerate(acts):
            path.append(sim.hand())
            now_closed = a[6] > 0
            if now_closed != closed and all((x[6] > 0) == now_closed for x in acts[t:t + 3]):
                events.append(dict(t=t, kind="grasp" if now_closed else "release", hand=sim.hand(),
                                   pos=sim.positions()))
                closed = now_closed
            _, _, _, _, info = env.step(a)
            if info.get("is_success"):
                success, end = True, t + 1
                break
        final = dict(t=end, hand=sim.hand(), pos=sim.positions())
        return dict(success=success, end=end, path=np.stack(path), events=events, final=final, start=start,
                    movable=sim.movable, goal=[list(g) for g in sim.inner.parsed_problem["goal_state"]])
    finally:
        env.close()


def anchor_body(arg, bodies):
    """The body a goal argument names (longest body name it starts with); else the argument itself, a fixed table
    region that no body carries (shift_vector then gives d = 0)."""
    cand = [b for b in bodies if arg == b or arg.startswith(b + "_")]
    return max(cand, key=len) if cand else arg


def phase_targets(rec, regrasp_dist=0.05):
    """[(start step, end step, body name)] for the source's phases, anchored by the BDDL goal."""
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
                name = carried  # let go and grasp it again where it was picked
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
    """Per-step d (xy only) for the source's path into the target scene."""
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


def target_start_positions(tid, init_index, seed):
    env = make_env(tid, init_index)
    try:
        env.reset(seed=int(seed))
        return Sim(env).positions()
    finally:
        env.close()


def job(args):
    (tid, k, seed), sources, kp, blend, modes = args
    tstart = target_start_positions(tid, k, seed)
    out = []
    for src in sources:
        rec = src["rec"]
        n = len(src["actions"])
        phases = phase_targets(rec)
        dvec, dinfo = shift_vector(rec, phases, tstart, n, blend)
        path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
        row = dict(target=tid, init=k, source=src["task_id"], source_cat=src["category"], phases=dinfo)
        for mode in modes:
            if mode == "replay":
                ok, steps = run_episode(tid, k, seed, src["actions"], "replay")
            elif mode == "track0":
                ok, steps = run_episode(tid, k, seed, src["actions"], "track", path, np.zeros((n, 3)), kp)
            else:
                ok, steps = run_episode(tid, k, seed, src["actions"], "track", path, dvec, kp)
            row[mode] = dict(success=ok, steps=steps)
        out.append(row)
    return out


def rec_job(src):
    rec = record_source(src)
    return src["key"], rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--per_target", type=int, default=3)
    ap.add_argument("--kp", type=float, default=1.0)
    ap.add_argument("--blend", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["replay", "track0", "trackd"])
    ap.add_argument("--limit", type=int, default=0, help="first N (target, init) jobs only (pilot)")
    ap.add_argument("--out", default="/root/pace/runs/rectest/v10_shift/oracle_shift2.json")
    args = ap.parse_args()

    # sources: one success per variant (light / noise / background) of every camera variant's base task
    eps = []
    for f in sorted(glob.glob(f"{RUN}/rollouts/rollouts_*.pt")):
        eps += torch.load(f, map_location="cpu", weights_only=False)["episodes"]
    rows = [json.loads(l) for l in open(f"{RUN}/episodes.jsonl")]
    cams = sorted({r["task_id"]: r for r in rows if r["category"] == "Camera Viewpoints"}.items())
    by_base = collections.defaultdict(dict)
    for e in eps:
        m = e["meta"]
        if e["success"] and m["category"] in SOURCE_CATS and m["task_id"] not in by_base[m["base"]]:
            by_base[m["base"]][m["task_id"]] = e
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.constants import ACTION

    cfg = PreTrainedConfig.from_pretrained("lerobot/pi05_libero_finetuned")
    _, policy_post = make_pre_post_processors(cfg, pretrained_path="lerobot/pi05_libero_finetuned",
                                              preprocessor_overrides={"device_processor": {"device": "cpu"},
                                                                      "rename_observations_processor": {"rename_map": {}}})
    _, env_post = make_env_pre_post_processors(LiberoPlusEnv(task="libero_10"), cfg)

    def post(a):  # the closed loop's action path: policy unnormalization, then the env postprocessor
        out = env_post({ACTION: policy_post(a)})[ACTION]
        return out if torch.is_tensor(out) else torch.as_tensor(np.asarray(out))
    sources = {}
    for base, d in by_base.items():
        order = sorted(d.values(), key=lambda e: (SOURCE_CATS.index(e["meta"]["category"]), e["meta"]["task_id"]))
        for e in order:
            m = e["meta"]
            acts = e["actions"][:, :10].reshape(-1, e["actions"].shape[-1]).float()
            env_acts = post(acts).float().cpu().numpy()  # the closed loop's unnormalization
            sources[(base, m["task_id"])] = dict(key=(base, m["task_id"]), task_id=int(m["task_id"]),
                                                 init_index=int(m.get("init_index", 0)), seed=int(m["seed"]),
                                                 category=m["category"], actions=env_acts.astype(np.float64))
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(args.workers, initializer=worker_init) as pool:
        recs = dict(pool.map(rec_job, list(sources.values())))
        bad = [k for k, r in recs.items() if not r["success"]]
        print(f"[sources] {len(recs)} replayed in their own env, {len(recs) - len(bad)} succeed again "
              f"(exact replay); dropped {bad} ({time.time() - t0:.0f}s)", flush=True)
        for k, r in recs.items():
            sources[k]["rec"] = r
        jobs = []
        for tid, r in cams:
            pool_src = [s for (b, _), s in sources.items() if b == r["base"] and s["rec"]["success"]][:args.per_target]
            for k in args.inits:
                jobs.append(((tid, k, 1000 + 7 * tid + k), pool_src, args.kp, args.blend, args.modes))
        if args.limit:
            jobs = jobs[:args.limit]
        print(f"[jobs] {len(jobs)} (target, init) x up to {args.per_target} sources x {args.modes}", flush=True)
        results = []
        for out in pool.imap_unordered(job, jobs):
            results += out
            json.dump(results, open(args.out, "w"), indent=1)
            done = len(results)
            print(f"[progress] {done} pairs ({time.time() - t0:.0f}s): " + ", ".join(
                f"{m} {sum(r[m]['success'] for r in results)}" for m in args.modes), flush=True)

    print("\n[summary] success over (target, init, source) pairs; by target variant:")
    for m in args.modes:
        s = sum(r[m]["success"] for r in results)
        print(f"  {m:7s} {s}/{len(results)} ({100 * s / max(len(results), 1):.0f}%)")
    if "trackd" in args.modes and "replay" in args.modes:
        tids = sorted({r["target"] for r in results})
        by = collections.defaultdict(list)
        for r in results:
            by[r["target"]].append(int(r["trackd"]["success"]) - int(r["replay"]["success"]))
        rng, ds = random.Random(0), []
        for _ in range(4000):
            smp = [rng.choice(tids) for _ in tids]
            v = [x for t in smp for x in by[t]]
            ds.append(100 * sum(v) / len(v))
        ds.sort()
        print(f"  trackd - replay: {100 * np.mean([x for v in by.values() for x in v]):+.0f} pp "
              f"[{ds[100]:+.0f}, {ds[3899]:+.0f}] (bootstrap over target variants)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
