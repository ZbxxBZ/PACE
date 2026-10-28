"""
V11 label pilot (10-27, user approved; desktop design doc 10.14c): how large is "pi0.5 seeing clearly - pi0.5 seeing the
perturbed view" at the same state? Same expert-pair samples as the signal check (runs/v11/check/check.npz, 240
episodes). Each episode is replayed (deterministic, render_pairs.replay) twice more:
  cam : the stored scene augmentation with the agentview camera move removed (cam_dpos 0, cam_angle 0); lights,
        colours, pixel noise unchanged (same per-frame noise draws) -> only the third-person camera differs, the
        wrist frames come out identical to the stored ones (checked);
  all : no augmentation at all.
pi0.5 then runs on the stored (perturbed) frames and on both clean versions with the SAME noise seed as the check's
first sample (1_000_003 * i + 1), plus the perturbed frames with the check's second seed (sampling spread), all in one
batch. Nothing rendered is stored; the chunks go to runs/v11/pilot/pilot.npz.
Reported (first 10 steps, normalized): |a_cam - a_pert|, |a_all - a_pert| vs |a_pert(seed 2) - a_pert(seed 1)|, by
dimension and by the size of the camera move; and whether seeing clearly moves pi0.5 toward a*.
    python v11_clean_pilot.py --workers 8 --out runs/v11/pilot
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import time

import numpy as np
import torch

V = "/root/pace/runs/v10b"
CALL, EXEC = 10, 10
GROUPS = (("all", slice(0, 7)), ("pos", slice(0, 3)), ("rot", slice(3, 6)), ("grip", slice(6, 7)))


def render_job(job):
    """(key, task, init, seed, aug) -> key, {variant: rec} for the camera-free and the augmentation-free replays."""
    import render_pairs as R

    key, tid, k, seed, aug = job
    with np.load(f"{V}/pairs/expert/{key}.npz") as z:
        acts = np.array(z["actions"], np.float64)
    out = {}
    for name, a in (("cam", dict(aug, cam_dpos=[0.0, 0.0, 0.0], cam_angle=0.0)), ("all", None)):
        ok, _, rec = R.replay(tid, k, seed, acts, a)
        out[name] = dict(rec, success=ok)
    return key, out


def rms(x, sl):
    return np.sqrt((x[:, :EXEC, sl] ** 2).mean((1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8, help="samples per forward (x 4 rows)")
    ap.add_argument("--out", default="/root/pace/runs/v11/pilot")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import collect_pairs as P
    from lerobot.envs.utils import preprocess_observation
    from v11_check import load_pi05, raw_obs

    C = np.load("/root/pace/runs/v11/check/check.npz", allow_pickle=True)
    kind, keys, ts = C["kind"], C["key"], C["t"].astype(int)
    ex = np.where(kind == "expert")[0]
    by_ep = collections.defaultdict(list)
    for g in ex:
        by_ep[str(keys[g])].append(int(g))
    pairs = {r["key"]: r for r in map(json.loads, open(f"{V}/pairs/expert.jsonl"))}
    lang = {r["task_id"]: r["language"] for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))}
    jobs = []
    for key in by_ep:
        with np.load(f"{V}/frames/expert/{key}_a0.npz") as z:
            aug = json.loads(str(z["aug"]))
        r = pairs[key]
        jobs.append((key, r["task_id"], r["init"], r["seed"], aug))
    print(f"[pilot] {len(jobs)} episodes, {len(ex)} samples", flush=True)

    policy, noise, pre, env_pre, _, _ = load_pi05()
    N = len(C["kind"])
    res = {v: np.full((N, 50, 7), np.nan, np.float32) for v in ("pert1", "pert2", "cam", "all")}
    wrist_same, succ_changed, t0 = [], 0, time.time()
    pending = []  # (global index, {variant: (frames, row)})

    def flush(items):
        rows, langs, seeds = [], [], []
        for v, sd in (("pert", 1), ("cam", 1), ("all", 1), ("pert", 2)):
            for g, src in items:
                rows.append(src[v])
                langs.append(src["lang"])
                seeds.append(1_000_003 * g + sd)
        obs = preprocess_observation(raw_obs(rows))
        obs["task"] = langs
        obs = pre(env_pre(obs))
        noise.set(seeds)
        with torch.inference_mode():
            a = policy.predict_action_chunk(obs).float().cpu().numpy()
        n = len(items)
        for j, (g, _) in enumerate(items):
            res["pert1"][g], res["cam"][g], res["all"][g], res["pert2"][g] = a[j], a[n + j], a[2 * n + j], a[3 * n + j]

    with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:
        for n_ep, (key, out) in enumerate(pool.imap_unordered(render_job, jobs)):
            with np.load(f"{V}/frames/expert/{key}_a0.npz") as z:
                orig = {k: np.array(z[k]) for k in ("agent", "wrist", "eef_pos", "eef_quat", "gripper", "joints", "t")}
            for name in ("cam", "all"):
                assert np.array_equal(out[name]["t"], orig["t"]), f"{key} {name}: replayed rows differ"
                succ_changed += int(not out[name]["success"])
            wrist_same.append(bool(np.array_equal(out["cam"]["wrist"], orig["wrist"])))
            row_of = {int(t): i for i, t in enumerate(orig["t"])}
            task = pairs[key]["task_id"]
            for g in by_ep[key]:
                i = row_of[int(ts[g])]
                cam = dict(orig, agent=out["cam"]["agent"])
                alls = dict(orig, agent=out["all"]["agent"], wrist=out["all"]["wrist"])
                pending.append((g, dict(pert=(orig, i), cam=(cam, i), all=(alls, i), lang=lang[task])))
                if len(pending) == args.batch:
                    flush(pending)
                    pending = []
            if n_ep % 20 == 0:
                print(f"[pilot] {n_ep + 1}/{len(jobs)} episodes ({time.time() - t0:.0f}s)", flush=True)
    if pending:
        flush(pending)

    np.savez(f"{args.out}/pilot.npz", **{k: v.astype(np.float16) for k, v in res.items()}, index=ex)
    a1, a2, ac, aa = (res[k][ex].astype(np.float64) for k in ("pert1", "pert2", "cam", "all"))
    tgt = C["target"][ex].astype(np.float64)
    chk = C["a_pi"][ex].astype(np.float64)
    lines = [f"{len(ex)} samples / {len(jobs)} episodes; wrist frames of the camera-free replay identical to the stored "
             f"ones: {sum(wrist_same)}/{len(wrist_same)}; replays that no longer succeed: {succ_changed} (expected 0)",
             f"reproducibility: |a_pert(seed 1) here - the check's a_pi| = {rms(a1 - chk, slice(0, 7)).mean():.4f}", "",
             "(first 10 steps, normalized)          all     pos     rot    grip"]
    for lab, x in (("two samples (seed 2 - seed 1)", a2 - a1), ("camera reset   a_cam - a_pert", ac - a1),
                   ("no augmentation a_all - a_pert", aa - a1), ("current label  a* - a_pert", tgt - a1),
                   ("a* - a_cam (clear view)", tgt - ac), ("a* - a_all", tgt - aa)):
        lines.append(f"  {lab:34s} " + "  ".join(f"{rms(x, sl).mean():.3f}" for _, sl in GROUPS))
    sp = rms(a2 - a1, slice(0, 7)).mean()
    lines.append(f"  ratio to the sampling spread: camera reset {rms(ac - a1, slice(0, 7)).mean() / sp:.2f}, "
                 f"no augmentation {rms(aa - a1, slice(0, 7)).mean() / sp:.2f}")
    ang, pos = C["cam_angle"][ex].astype(float), C["cam_pos"][ex].astype(float)
    for nm, v, edges in (("camera angle deg", ang, (0, 4, 8, 12.1)), ("camera move m", pos, (0, 0.06, 0.1, 0.2))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (v >= lo) & (v < hi)
            if s.any():
                lines.append(f"  by {nm} [{lo}, {hi}): n {s.sum():5d}  a_cam - a_pert {rms((ac - a1)[s], slice(0, 7)).mean():.3f}"
                             f" (pos {rms((ac - a1)[s], slice(0, 3)).mean():.3f})  spread {rms((a2 - a1)[s], slice(0, 7)).mean():.3f}")
    big = rms(ac - a1, slice(0, 7)) > 3 * rms(a2 - a1, slice(0, 7))
    lines.append(f"  samples where the camera reset changes pi0.5 by > 3x its own spread: {100 * big.mean():.1f}%")
    txt = "\n".join(lines)
    open(f"{args.out}/summary.txt", "w").write(txt + "\n")
    print(txt, flush=True)


if __name__ == "__main__":
    main()
