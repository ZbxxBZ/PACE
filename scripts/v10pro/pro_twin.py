"""
Section 7.4 of docs/PACE_V10_经验配准方案.md (registered 10-23 18:20): counterfactual observation -- pi0.5 looks at a
twin of the scene in which the current phase's anchor is back where it remembers it. Upper bound with simulator truth on
libero_10_swap (10 tasks x init states 0..9, the base run's seeds).
Twin = the original libero_10 scene with the same init state index (fixtures where pi0.5 knows them). Before every
query: objects the robot has not moved yet (real position within 1 cm of where it started) stay where pi0.5 knows
them -- the current anchor is there by definition of d; objects it has moved (carried or placed) are copied from the
real scene shifted by -d; every fixture joint (drawer, door, knob) copied; the twin arm put by damped least-squares IK so its hand is at p - d with the real hand's orientation, the
gripper fingers copied; the twin's two cameras and robot state go to pi0.5; its actions are executed in the real scene.
Phases and d as in pro_shift.py: anchors from the task's first experience, d = live anchor position (swap) - its
position in the original scene (xy); the phase advances on pi0.5's gripper change (3 steps). 520 steps.
    python pro_twin.py --workers 10 -> runs/rectest/v10_pro/twin_l10_swap.json
"""
import argparse
import json
import multiprocessing as mp
import random
import socket
import time

import mujoco
import numpy as np

import pro_common as C
import pro_oracle as P
from pro_wire import recv_msg, send_msg

OUT = "/root/pace/runs/rectest/v10_pro/twin_l10_swap.json"


def rotvec(R):
    """Rotation vector of a rotation matrix."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-8:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return th * v


class Twin:
    def __init__(self, task_name, init_idx, seed):
        idx = P.task_index("libero_10")[task_name]
        self.env, _, inits = C.make_env("libero_10", idx, C.SIZE)
        C.reset(self.env, inits[init_idx], seed)
        e = self.env.env
        self.e, self.sim, self.robot = e, e.sim, e.robots[0]
        self.known = P.Sim(self.env).positions()
        self.m, self.d = self.sim.model._model, self.sim.data._data
        self.q0 = np.array(self.sim.data.qpos)  # the familiar layout
        self.real0 = None                         # the real scene's object qpos at the start

    def addr(self, sim, joint):
        a = sim.model.get_joint_qpos_addr(joint)
        return (a, a + 1) if isinstance(a, (int, np.integer)) else a

    def sync(self, real_env, dvec):
        """Copy the real scene into the twin, movable objects shifted by -d, hand at p - d. Returns the IK error (m)."""
        re, rs = real_env.env, real_env.env.sim
        q = self.sim.data.qpos
        if self.real0 is None:
            self.real0 = np.array(rs.data.qpos)
        for name, obj in re.objects_dict.items():
            if name not in self.e.objects_dict:
                continue
            for j in obj.joints:
                a0, a1 = self.addr(rs, j)
                b0, b1 = self.addr(self.sim, j)
                val = np.array(rs.data.qpos[a0:a1])
                if a1 - a0 == 7 and np.linalg.norm(val[:3] - self.real0[a0:a0 + 3]) < 0.01:
                    q[b0:b1] = self.q0[b0:b1]  # not moved by the robot yet: where pi0.5 knows it
                    continue
                if a1 - a0 == 7:
                    val[:2] -= dvec[:2]
                q[b0:b1] = val
        for name, fx in re.fixtures_dict.items():
            for j in (getattr(fx, "joints", None) or []):
                try:
                    a0, a1 = self.addr(rs, j)
                    b0, b1 = self.addr(self.sim, j)
                    if a1 - a0 < 7:
                        q[b0:b1] = rs.data.qpos[a0:a1]
                except Exception:
                    pass
        r = re.robots[0]
        q[self.robot._ref_gripper_joint_pos_indexes] = rs.data.qpos[r._ref_gripper_joint_pos_indexes]
        arm = self.robot._ref_joint_pos_indexes
        q[arm] = rs.data.qpos[r._ref_joint_pos_indexes]
        self.sim.data.qvel[:] = 0
        sid = self.robot.eef_site_id
        tgt_p = np.array(rs.data.site_xpos[r.eef_site_id]) - dvec
        tgt_R = np.array(rs.data.site_xmat[r.eef_site_id]).reshape(3, 3)
        dofs = self.robot._ref_joint_vel_indexes
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        err = 1.0
        for _ in range(100):
            self.sim.forward()
            p = np.array(self.sim.data.site_xpos[sid])
            R = np.array(self.sim.data.site_xmat[sid]).reshape(3, 3)
            e6 = np.concatenate([tgt_p - p, rotvec(tgt_R @ R.T)])
            err = float(np.linalg.norm(e6[:3]))
            if err < 1e-4 and np.linalg.norm(e6[3:]) < 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, sid)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e6)
            q[arm] = q[arm] + np.clip(dq, -0.2, 0.2)
        self.sim.forward()
        return err

    def observation(self):
        return self.e._get_observations(force_update=True)

    def close(self):
        self.env.close()


def run_cell(job):
    (t, k), plan, port = job
    seed = 1000 + 100 * t + k
    env, task, inits = C.make_env("libero_10_swap", t, C.SIZE)
    twin = Twin(task.name, k, seed)
    sock = socket.create_connection(("127.0.0.1", port))
    try:
        obs = C.reset(env, inits[k], seed)
        sim = P.Sim(env)
        phase, run, chunk, success, steps, calls, log = 0, 0, None, False, C.max_steps("libero_10_swap"), 0, []
        for step in range(C.max_steps("libero_10_swap")):
            if step % 10 == 0:
                kind, anchor = plan[phase]
                d = np.zeros(3)
                if anchor in twin.known:
                    d[:2] = sim.positions()[anchor][:2] - twin.known[anchor][:2]
                ik = twin.sync(env, d)
                tobs = twin.observation()
                send_msg(sock, {"task": task.language, "seed": seed, "call": calls}, C.raw_fields(twin.env, tobs))
                _, a = recv_msg(sock)
                chunk = a["actions"]
                calls += 1
                log.append([phase, round(float(np.hypot(d[0], d[1])), 3), round(ik, 4)])
            act = np.asarray(chunk[step % 10], dtype=np.float64)
            obs, _, _, _ = env.step(act)
            if env.check_success():
                success, steps = True, step + 1
                break
            nxt = plan[phase][0]
            if nxt is not None and phase + 1 < len(plan):
                want = 1.0 if nxt == "grasp" else -1.0
                run = run + 1 if np.sign(act[6]) == want else 0
                if run >= 3:
                    phase, run = phase + 1, 0
        return dict(task=t, init=k, success=success, steps=steps, phase_reached=phase, n_phases=len(plan),
                    max_ik_err=max(x[2] for x in log), log=log)
    finally:
        sock.close()
        twin.close()
        env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    swap = P.task_index("libero_10_swap")
    names = {i: n for n, i in swap.items()}
    plans = {}
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        if r["success"] and r["name"] not in plans and swap[r["name"]] in args.tasks:
            key = "t%d_i%d" % (r["task"], r["init"])
            rec = P.record_source(dict(key=key, task=r["task"], init=r["init"], seed=r["seed"], actions=arrays[key]))
            assert rec["success"], key
            evs = rec["events"]
            plans[r["name"]] = [(evs[i]["kind"] if i < len(evs) else None, a) for i, (_, _, a) in
                                enumerate(P.phase_targets(rec))]
    jobs = [((t, k), plans[names[t]], args.port) for t in args.tasks for k in args.inits]
    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)
            print(f"[progress] {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s): twin "
                  f"{sum(x['success'] for x in results)}/{len(results)}", flush=True)
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    sel = {(r["task"], r["init"]): r["success"] for r in results}
    tasks = sorted({c[0] for c in sel})
    rng = random.Random(0)
    diff = lambda ts: np.mean([sel[c] - base[c] for c in sel if c[0] in ts])
    bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
    summary = dict(base=round(100 * np.mean([base[c] for c in sel]), 1), twin=round(100 * np.mean(list(sel.values())), 1),
                   twin_minus_base=[round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)],
                   per_task={f"{t} {names[t][:55]}": dict(base=round(100 * np.mean([base[c] for c in sel if c[0] == t])),
                                                          twin=round(100 * np.mean([sel[c] for c in sel if c[0] == t])))
                             for t in tasks})
    json.dump(summary, open(args.out.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
