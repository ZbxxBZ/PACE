"""
Can frozen image features locate the phase's target to ~1 cm? (10-23, gate 2 of the registration plan; no training)
For the generalization cells (camera variants x init states, sources from collect_pi05.py noise_sources), follow the
source with the true d (expert) and every 10 steps render both cameras at 224 x 224. Template: in the source's frame
at the end of the phase (the grasp / release moment), the 3 x 3 patch features around the phase anchor's projection.
In the current frame, every 3 x 3 window is scored by the mean patch cosine; the soft-argmax around the best window
gives a pixel, back-projected onto the horizontal plane at the anchor's height in the source -> estimated anchor xy;
error = |estimate - true current anchor xy|. Reported by camera and by the hand's xy distance to the anchor.
Wrist camera: pose from the robot's kinematics (available on a real robot). Agentview: the CURRENT (perturbed) camera
pose from the simulator -- unknown at test time, so its numbers are optimistic.
Features: pi0.5's frozen SigLIP vision tower (patch tokens, 16 x 16 at 224). --check writes an annotated image to
verify the projection convention first.
    python v10_loc_probe.py --check ; python v10_loc_probe.py --workers 10 -> runs/rectest/v10_loc/result.json
Variants (10-23, after the first run: agentview median ~3.5 cm with 30-45 cm tails = matches on the wrong object):
  --radius R   only windows whose point on the anchor plane lies within R (m) of the source's anchor xy -- the network's
               d is clipped to +-6 cm anyway, so the search never needs to look further;
  LOC_RES=448 --zoom_cams agentview   render at 448 and match inside a 224 crop centred on the source anchor's
               projection in the current camera (2x zoom: ~3 cm per SigLIP patch on the table instead of ~6).
    LOC_RES=448 python v10_loc_probe.py --radius 0.08 --zoom_cams agentview --tag r8_z2
  --template aligned   template from the source's frame at the same step instead of the phase-end close-up (same scale
               as the current view far from the object).
  --offset E   the expert follows the true d plus E (m) in a random direction per phase. Without it the current view
               repeats the reference's view (the hand is already where it should be), so "same pixel" is always right
               and an aligned template scores ~0 trivially (first aligned run, 10-23: wrist 0.06 cm).
  --known_pose_cams robot0_eye_in_hand   the realistic case: the perturbed agentview's pose is unknown at test time, so
               its pixel is back-projected with the source (default) camera's pose.
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import time

import numpy as np

import v10_oracle_shift2 as O

OUT = "/root/pace/runs/rectest/v10_loc"
S = int(os.environ.get("LOC_RES", "224"))  # render resolution; features always see a 224 crop (--zoom = S / 224)
C = 224
CAMS = ("agentview", "robot0_eye_in_hand")


def cam_mats(sim, cam):
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_transform_matrix

    return get_camera_transform_matrix(sim, cam, S, S), get_camera_extrinsic_matrix(sim, cam)


def render(inner):
    o = inner._get_observations(force_update=True)
    return {c: np.ascontiguousarray(o[c + "_image"]) for c in CAMS}


def snapshot(inner, anchor_names):
    sim = inner.sim
    pos = {n: np.array(sim.data.body_xpos[inner.obj_body_id[n]], dtype=np.float64) for n in anchor_names
           if n in inner.obj_body_id}
    hand = np.array(sim.data.site_xpos[inner.robots[0].eef_site_id], dtype=np.float64)
    return dict(img=render(inner), mats={c: cam_mats(sim, c) for c in CAMS}, pos=pos, hand=hand)


def make224(tid, k):
    c = O.G["env_cfg"]
    fn = O.G["make_fns"](suite=O.G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1,
                         camera_names=O.G["cams"], episode_length=c.episode_length, init_states=c.init_states,
                         gym_kwargs=O.G["gk"], control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    kw = dict(fn.keywords)
    kw.update(observation_width=S, observation_height=S)
    return fn.func(int(k), **kw)


def source_templates(src, aligned=False):
    """Replay the source in its own scene; a snapshot at every phase's end event."""
    rec = src["rec"]
    phases = O.phase_targets(rec)
    ends = {e: b for (_, e, b) in phases}
    env = make224(src["task_id"], src["init_index"])
    try:
        env.reset(seed=int(src["seed"]))
        inner = env._env.env
        snaps = {}
        for t, a in enumerate(src["actions"]):
            if t in ends:
                snaps[t] = dict(snapshot(inner, [ends[t]]), anchor=ends[t])
            if aligned and t % 10 == 0:  # the reference frame at the same step (what the network's K/V window holds)
                ph = next(i for i, (s, e, _) in enumerate(phases) if t < max(e, s + 1) or i == len(phases) - 1)
                b = phases[ph][2]
                if b in inner.obj_body_id:
                    snaps[f"a{t}"] = dict(snapshot(inner, [b]), anchor=b)
            env.step(a)
        return snaps
    finally:
        env.close()


def job(args):
    (tid, k, seed), src, aligned, offset = args
    rec = src["rec"]
    n = len(src["actions"])
    phases = O.phase_targets(rec)
    tstart = O.target_start_positions(tid, k, seed)
    dvec, _ = O.shift_vector(rec, phases, tstart, n, 15)
    path = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
    templ = source_templates(src, aligned)
    rng = np.random.default_rng([int(tid), int(k)])
    off = [offset * np.array([np.cos(th), np.sin(th), 0.0]) for th in rng.uniform(0, 2 * np.pi, len(phases))]
    env = make224(tid, k)
    frames = []
    try:
        env.reset(seed=int(seed))
        inner = env._env.env
        sim = O.Sim(env)
        for t in range(n):
            ph = next(i for i, (s, e, _) in enumerate(phases) if t < max(e, s + 1) or i == len(phases) - 1)
            s_, e_, anchor = phases[ph]
            if t % 10 == 0 and e_ in templ and anchor in inner.obj_body_id:
                frames.append(dict(snapshot(inner, [anchor]), t=t, phase=ph, anchor=anchor, end=e_))
            a = np.array(src["actions"][t], dtype=np.float64)
            tgt = path[t] + dvec[t] + off[ph]
            a[:3] = np.clip(a[:3] + (tgt - sim.hand()) / 0.05, -1.0, 1.0)
            _, _, _, _, info = env.step(a)
            if info.get("is_success"):
                break
        return dict(target=tid, init=k, source=src["task_id"], templ=templ, frames=frames,
                    success=bool(info.get("is_success")))
    finally:
        env.close()


def project(T, p):
    """World point -> (row, col) in the rendered image array. robosuite's projection counts rows from the bottom of
    the image the renderer returns (checked against the depth map 10-23), so the row is flipped."""
    q = T @ np.append(p, 1.0)
    return np.array([S - 1 - q[1] / q[2], q[0] / q[2]])


def crop_box(rc):
    """Top-left (row, col) of the C x C crop centred on rc, kept inside the S x S image."""
    return (int(np.clip(round(rc[0]) - C // 2, 0, S - C)), int(np.clip(round(rc[1]) - C // 2, 0, S - C)))


def backproject(T, E, rc, z):
    """Pixel (row, col) -> the point on the plane z it sees (T: world -> pixel, E: camera -> world)."""
    Kc = T @ E  # camera -> pixel
    K = Kc[:3, :3]
    ray_c = np.linalg.solve(K, np.array([rc[1], S - 1 - rc[0], 1.0]))  # camera frame direction (row unflipped)
    R, o = E[:3, :3], E[:3, 3]
    ray_w = R @ ray_c
    if abs(ray_w[2]) < 1e-9:
        return None
    lam = (z - o[2]) / ray_w[2]
    return o + lam * ray_w if lam > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inits", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sources", default="/root/pace/runs/v10b/noise_sources")
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--radius", type=float, default=0.0)
    ap.add_argument("--zoom_cams", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--offset", type=float, default=0.0)
    ap.add_argument("--template", choices=["phase_end", "aligned"], default="phase_end")
    ap.add_argument("--known_pose_cams", nargs="*", default=list(CAMS))  # others use the source's camera pose
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(f"{args.sources}/episodes.jsonl")]
    srcs = {}
    for r in rows:
        if r["success"] and r["base"] not in srcs:  # one source per base task
            srcs[r["base"]] = dict(key=(r["base"], r["task_id"]), task_id=int(r["task_id"]), init_index=int(r["init"]),
                                   seed=int(r["seed"]), category=r["category"],
                                   actions=np.load(f"{args.sources}/{r['actions']}").astype(np.float64))
    targets = [json.loads(l) for l in open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl")]
    cams = sorted({r["task_id"]: r["name"].split("_view_")[0] for r in targets}.items())
    O.worker_init()
    for b, s in srcs.items():
        s["rec"] = O.rec_job(s)[1]
    jobs = [((tid, k, 1000 + 7 * tid + k), srcs[b], args.template == "aligned", args.offset) for tid, b in cams if b in srcs for k in args.inits]
    if args.check:
        import cv2

        out = job(jobs[0])
        f = out["frames"][min(3, len(out["frames"]) - 1)]
        tiles = []
        for c in CAMS:
            T, _ = f["mats"][c]
            rc = project(T, f["pos"][f["anchor"]])
            im = cv2.resize(np.ascontiguousarray(f["img"][c]), (S, S)).copy()
            cv2.circle(im, (int(rc[1]), int(rc[0])), 6, (255, 0, 0), 2)
            tiles.append(im)
        cv2.imwrite(f"{OUT}/check.png", cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))
        print("CHECK anchor", f["anchor"], "pixels", [project(f["mats"][c][0], f["pos"][f["anchor"]]).round(1).tolist()
                                                     for c in CAMS], "->", f"{OUT}/check.png", flush=True)
        return
    t0 = time.time()
    import torch

    from pace.common import load_policy

    policy, _ = load_policy("lerobot/pi05_libero_finetuned", "cuda", n_action_steps=10)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower

    @torch.no_grad()
    def feats(img):  # uint8 HxWx3 -> [16, 16, D] unit vectors (SigLIP: pixels in [-1, 1])
        x = torch.from_numpy(img).float().permute(2, 0, 1)[None].cuda() / 127.5 - 1.0
        h = vt(pixel_values=x.to(next(vt.parameters()).dtype)).last_hidden_state[0].float()
        h = torch.nn.functional.normalize(h, dim=-1)
        g = int(round(h.shape[0] ** 0.5))
        return h.reshape(g, g, -1)

    res = collections.defaultdict(list)
    with mp.get_context("spawn").Pool(args.workers, initializer=O.worker_init) as pool:
        for out in pool.imap_unordered(job, jobs):
            for f in out["frames"]:
                tp = out["templ"].get(f"a{f['t']}") if args.template == "aligned" else None
                tp = tp or out["templ"][f["end"]]
                dist = float(np.linalg.norm(f["hand"][:2] - f["pos"][f["anchor"]][:2]))
                for c in CAMS:
                    T0, _ = tp["mats"][c]
                    T1, E1 = f["mats"][c] if c in args.known_pose_cams else tp["mats"][c]
                    a0 = tp["pos"][tp["anchor"]]
                    rc0 = project(T0, a0)
                    zoom = c in args.zoom_cams
                    b0 = crop_box(rc0) if zoom else (0, 0)
                    b1 = crop_box(project(T1, a0)) if zoom else (0, 0)  # prior: where the source's anchor would be
                    img0, img1 = tp["img"][c], f["img"][c]
                    if not zoom and S != C:
                        import cv2
                        img0, img1 = (cv2.resize(np.ascontiguousarray(x), (C, C), interpolation=cv2.INTER_AREA)
                                      for x in (img0, img1))
                    sc_ = 1.0 if zoom else S / C  # crop pixel -> full-image pixel
                    F0 = feats(np.ascontiguousarray(img0[b0[0]:b0[0] + C, b0[1]:b0[1] + C]))
                    F1 = feats(np.ascontiguousarray(img1[b1[0]:b1[0] + C, b1[1]:b1[1] + C]))
                    g = F0.shape[0]
                    p = C / g
                    r0c = (rc0 - np.array(b0)) / sc_  # template anchor inside its crop
                    i0, j0 = int(np.clip(r0c[0] // p, 1, g - 2)), int(np.clip(r0c[1] // p, 1, g - 2))
                    off = np.array([r0c[0] - (i0 + 0.5) * p, r0c[1] - (j0 + 0.5) * p])
                    tmpl = F0[i0 - 1:i0 + 2, j0 - 1:j0 + 2]  # [3, 3, D]
                    sc = torch.full((g, g), -2.0, device=F1.device)
                    for i in range(1, g - 1):
                        for j in range(1, g - 1):
                            if args.radius > 0:
                                q = backproject(T1, E1, np.array(b1) + sc_ * (np.array([(i + 0.5) * p, (j + 0.5) * p])
                                                                             + off), a0[2])
                                if q is None or np.linalg.norm(q[:2] - a0[:2]) > args.radius:
                                    continue
                            sc[i, j] = (F1[i - 1:i + 2, j - 1:j + 2] * tmpl).sum(-1).mean()
                    bi, bj = divmod(int(torch.argmax(sc)), g)
                    lo_i, hi_i, lo_j, hi_j = max(bi - 1, 1), min(bi + 2, g - 1), max(bj - 1, 1), min(bj + 2, g - 1)
                    loc = sc[lo_i:hi_i, lo_j:hi_j]
                    w = torch.softmax(loc.flatten() / args.temp, 0).reshape(loc.shape).cpu().numpy()
                    ii, jj = np.meshgrid(np.arange(lo_i, hi_i), np.arange(lo_j, hi_j), indexing="ij")
                    rcc = np.array([((ii + 0.5) * w).sum() * p, ((jj + 0.5) * w).sum() * p]) + off
                    rc = np.array(b1) + sc_ * rcc
                    est = backproject(T1, E1, rc, a0[2])
                    vis = all(0 <= v < S for v in project(f["mats"][c][0], f["pos"][f["anchor"]]))
                    err = float(np.linalg.norm(est[:2] - f["pos"][f["anchor"]][:2])) if est is not None else 1.0
                    res[c].append(dict(dist=dist, err=err, visible=vis, peak=float(sc.max())))
            print(f"[progress] {sum(len(v) for v in res.values()) // 2} frames ({time.time() - t0:.0f}s)", flush=True)
    name = f"result{'_' + args.tag if args.tag else ''}.json"
    json.dump(res, open(f"{OUT}/{name}", "w"))
    bins = [(0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 1.0)]
    print("\n[result] median / 75th / 90th percentile localization error (cm), anchor visible in the camera; n")
    for c in CAMS:
        print(f"-- {c}")
        for lo, hi in bins:
            e = np.array([r["err"] for r in res[c] if lo <= r["dist"] < hi and r["visible"]])
            if len(e):
                print(f"  hand {100 * lo:4.0f}-{100 * hi:<4.0f} cm: {100 * np.median(e):5.2f} / "
                      f"{100 * np.percentile(e, 75):5.2f} / {100 * np.percentile(e, 90):5.2f}  n={len(e)}")
    print(f"-> {OUT}/{name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
