"""
Route B (10-23): "stage + pi0.5" with two changes found by stage_diag.py. After staging, pi0.5 pulled the hand back
toward where it remembers the object (task 9: 3.8 cm, exactly the swap) and grasped air. So during pi0.5's contact:
  --xy_gain K : per-dimension blend -- pi0.5 keeps z, rotation and gripper; the xy command gets K * (target_xy -
                hand_xy) / 0.05 added (target = the experience's hand at the contact + d, clipped to [-1, 1]); K = 0 is pro_stage.py.
  --retries R : after a grasp phase, if the gripper closed on nothing (finger opening below EMPTY), open, re-stage and
                let pi0.5 grasp again, up to R times. The gripper opening is available on a real robot too.
Everything else as pro_stage.py (100 cells libero_10_swap, base seeds, first experience, staging, 15-step settle, 150
steps per pi0.5 phase, 520 steps). Upper bound with simulator truth for the staging target.
    python pro_stage2.py --xy_gain 1 --retries 2 -> runs/rectest/v10_pro/stage2_k<K>_r<R>_l10_swap.json
"""
import argparse
import json
import multiprocessing as mp
import random
import time

import numpy as np

import pro_common as C
import pro_oracle as P
import pro_stage as S

EMPTY = 0.004  # finger opening after closing: ~0.002 on nothing, >= 0.006 holding an object (pilot 10-23)


def finger_gap(ep):
    return float(np.sum(np.abs(ep.obs["robot0_gripper_qpos"])))


def vla_until(ep, kind, target, k_xy):
    want = 1.0 if kind == "grasp" else -1.0
    run, chunk = 0, None
    for i in range(S.VLA_PHASE_STEPS if kind else S.BUDGET):
        if i % 10 == 0:
            chunk = ep.client.act(ep.env, ep.obs, ep.task.language, ep.seed, ep.calls)
            ep.calls += 1
        a = np.array(chunk[i % 10], dtype=np.float64)
        if k_xy > 0 and target is not None:
            a[:2] = np.clip(a[:2] + k_xy * (target[:2] - ep.sim.hand()[:2]) / 0.05, -1.0, 1.0)
        if ep.step(a):
            return True
        run = run + 1 if (kind and np.sign(a[6]) == want) else 0
        if kind and run >= 3:
            return False
    return False


def settle(ep, grip):
    for _ in range(S.GRIP_SETTLE):
        a = np.zeros(7)
        a[6] = grip
        if ep.step(a):
            return True
    return False


def run_cell(job):
    (t, k), src, port, k_xy, retries = job
    seed = 1000 + 100 * t + k
    client = C.PolicyClient(port)
    ep = S.Episode(t, k, seed, client)
    rec = src["rec"]
    path = rec["path"]
    z_safe = float(path[:, 2].max())
    reached, tries, gaps = 0, 0, []
    try:
        done = False
        for ph in S.plan(rec):
            for attempt in range(retries + 1):
                d = np.zeros(3)
                if ph["anchor_pos"] is not None:
                    d[:2] = ep.sim.positions()[ph["anchor"]][:2] - ph["anchor_pos"][:2]
                target = path[min(ph["pre"], len(path) - 1)] + d
                if ep.stage(target, z_safe):
                    done = True
                    break
                contact = path[min(ph["t"], len(path) - 1)] + d  # the experience's hand at the contact, shifted
                done = vla_until(ep, ph["kind"], contact if ph["kind"] else None, k_xy)
                if done:
                    break
                done = settle(ep, ep.grip)
                if ph["kind"] == "grasp":
                    gaps.append(round(finger_gap(ep), 4))
                if done or ph["kind"] != "grasp" or finger_gap(ep) >= EMPTY or attempt == retries:
                    break
                tries += 1  # closed on nothing: open and try again
                done = settle(ep, -1.0)
                if done:
                    break
            reached += 1
            if done:
                break
        if not ep.success and ep.t < S.BUDGET:
            vla_until(ep, None, None, 0.0)
        return dict(task=t, init=k, success=ep.success, steps=ep.t, phases=len(S.plan(rec)), reached=reached,
                    retries=tries, gaps=gaps)
    finally:
        ep.close()
        client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--xy_gain", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or "/root/pace/runs/rectest/v10_pro/stage2_k%g_r%d_l10_swap.json" % (args.xy_gain, args.retries)
    rows = [json.loads(l) for l in open(f"{P.EXP}/episodes.jsonl")]
    arrays = np.load(f"{P.EXP}/actions.npz")
    swap = P.task_index("libero_10_swap")
    names = {i: n for n, i in swap.items()}
    first = {}
    for r in sorted(rows, key=lambda r: (r["task"], r["init"])):
        if r["success"] and r["name"] not in first and swap[r["name"]] in args.tasks:
            first[r["name"]] = dict(key=f"t{r['task']}_i{r['init']}", task=r["task"], init=r["init"], seed=r["seed"],
                                    actions=arrays[f"t{r['task']}_i{r['init']}"])
    for name, src in first.items():
        src["rec"] = P.record_source(src)
        assert src["rec"]["success"], name
    jobs = [((t, k), first[names[t]], args.port, args.xy_gain, args.retries) for t in args.tasks for k in args.inits]
    t0, results = time.time(), []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            results.append(r)
            json.dump(results, open(out, "w"), indent=1)
            print(f"[progress] {len(results)}/{len(jobs)} ({time.time() - t0:.0f}s): "
                  f"{sum(x['success'] for x in results)}/{len(results)}", flush=True)
    base = {(r["task"], r["init"]): r["success"]
            for r in map(json.loads, open("/root/pace/runs/rectest/v10_pro/base_l10_swap/episodes.jsonl"))}
    sel = {(r["task"], r["init"]): r["success"] for r in results}
    tasks = sorted({c[0] for c in sel})
    rng = random.Random(0)
    diff = lambda ts: np.mean([sel[c] - base[c] for c in sel if c[0] in ts])
    bs = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(4000))
    summary = dict(xy_gain=args.xy_gain, retries=args.retries, base=round(100 * np.mean([base[c] for c in sel]), 1),
                   stage2=round(100 * np.mean(list(sel.values())), 1),
                   minus_base=[round(100 * diff(tasks), 1), round(100 * bs[100], 1), round(100 * bs[3899], 1)],
                   per_task={f"{t} {names[t][:55]}": dict(base=round(100 * np.mean([base[c] for c in sel if c[0] == t])),
                                                          stage2=round(100 * np.mean([sel[c] for c in sel if c[0] == t])))
                             for t in tasks})
    json.dump(summary, open(out.replace(".json", "_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
