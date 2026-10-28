"""
V11 step 1 (10-27; desktop design doc 10.14c, docs/PACE_V1-V8.md "V11"): how large is the residual the V11 branch would
have to learn, a* - a_pi, on the training states, compared with pi0.5's own sampling spread?
  expert  : V10's expert pairs (reference + true-d controller, success only; runs/v10b/pairs/expert). At every pi0.5
            call (t % 10 == 0) of the rendered pair (runs/v10b/frames/expert/<key>_a0.npz: both cameras at 224 under
            the pair's scene augmentation, which moves the agentview camera by up to 10 cm / 12 deg) pi0.5 is run offline
            with two noise seeds: a_pi (seed 1), a_pi' (seed 2). a* = the executed expert actions t .. t + 49 (after the
            end: hold, last gripper command). Distances in pi0.5's normalized action space (MEAN_STD).
  pi05_aug: input check -- pi0.5's own rollouts under the same augmentations (runs/v10b/pi05_aug, frames/pi05_aug):
            offline a_pi vs the actions pi0.5 actually executed then should be about as far apart as two samples.
Offline inputs go through lerobot's own pipeline (preprocess_observation -> LIBERO env processor -> policy
preprocessor), from the env-format observation the frames were taken from (render_pairs.py stores the 224 env's
observation images and robot0_eef_pos / eef_quat / gripper_qpos / joint_pos). Closed loop renders at 360 and pi0.5
resizes to 224; the frames are native 224 renders -- the pi05_aug check measures what that changes.
    python v11_check.py --n_expert 240 --n_aug 60 --out runs/v11/check
--dump KIND [KIND ...] (V11 training data, run after the check passes): pi0.5's chunk at EVERY call row of every rendered
pair episode of those kinds, one noise seed per (episode, row), saved per episode as runs/v11/api/<kind>/<fkey>.npy
([frame rows, 50, 7] float16, normalized; NaN on rows that are not pi0.5 calls) plus runs/v11/api/norm.npz (mean, std).
    python v11_check.py --dump expert negative --out runs/v11/api
--refvis (V11, the V9-structure branch): every reference's pi0.5 image tokens at every frame row, as the V9 v2 encoder
reads them (pace.pi05.vis_tokens: per camera a 4 x 4 average-pooled grid of pi0.5's image embeddings, unit cells, both
cameras -> [rows, 32, 2048]) and its preprocessed robot state (normalized as pi0.5's input), from the references' 224
frames (augmentation 0) -> runs/v11/refvis/<key>_a0.npz (vis float16, state, t).
    python v11_check.py --refvis --out runs/v11/refvis
"""
import argparse
import json
import os
import time

import numpy as np
import torch

V = "/root/pace/runs/v10b"
CHUNK, EXEC, CALL = 50, 10, 10
CAMS = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}  # lerobot's LIBERO default mapping


class RowNoise:
    """pi0.5's flow-matching noise per batch row from a generator seeded by that row's seed (closed_loop_v3.py)."""

    def __init__(self):
        self.gens = None

    def set(self, seeds):
        self.gens = [torch.Generator().manual_seed(int(s)) for s in seeds]

    def __call__(self, shape, device):
        assert self.gens is not None and shape[0] == len(self.gens), (shape, len(self.gens or []))
        return torch.stack([torch.randn(tuple(shape[1:]), generator=g) for g in self.gens]).to(device)


def target_chunk(actions, t):
    """a*: the executed actions t .. t + CHUNK - 1; past the end: hold still with the last gripper command."""
    n = len(actions)
    return np.stack([actions[t + k] if t + k < n else np.r_[np.zeros(6), actions[-1][6]] for k in range(CHUNK)])


def raw_obs(rows):
    """The vector env's observation format (batched numpy) for frame rows (frames npz, row index)."""
    b = len(rows)
    cat = lambda k: np.stack([fr[k][i] for fr, i in rows])
    return {"pixels": {CAMS["agentview_image"]: cat("agent"), CAMS["robot0_eye_in_hand_image"]: cat("wrist")},
            "robot_state": {"eef": {"pos": cat("eef_pos"), "quat": cat("eef_quat"), "mat": np.zeros((b, 3, 3))},
                            "gripper": {"qpos": cat("gripper"), "qvel": np.zeros((b, 2))},
                            "joints": {"pos": cat("joints"), "vel": np.zeros((b, 7))}}}


def dist(x, y):
    """Per sample: RMS over (steps, 7 dims) and mean L2 of the position dims, on the executed part and the chunk."""
    d = x - y
    return dict(rms_exec=np.sqrt((d[:, :EXEC] ** 2).mean((1, 2))), rms_all=np.sqrt((d ** 2).mean((1, 2))),
                pos_exec=np.linalg.norm(d[:, :EXEC, :3], axis=2).mean(1), pos_all=np.linalg.norm(d[:, :, :3], axis=2).mean(1))


def load_pi05():
    """Frozen pi0.5 with per-row noise, its processors, and the action normalization (mean, std)."""
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from pace.common import load_policy

    policy, _ = load_policy("lerobot/pi05_libero_finetuned", "cuda", n_action_steps=10)
    noise = RowNoise()
    policy.model.sample_noise = noise
    pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                         preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                                 "rename_observations_processor": {"rename_map": {}}})
    env_pre, _ = make_env_pre_post_processors(LiberoPlusEnv(task="libero_10"), policy.config)
    st = next(s.stats for s in post.steps if getattr(s, "stats", None))["action"]
    return policy, noise, pre, env_pre, np.asarray(st["mean"], np.float64), np.asarray(st["std"], np.float64)


def dump(args, lang, preprocess_observation):
    """--dump: pi0.5's chunk at every call row of every rendered pair episode of the given kinds (resumable)."""
    import zlib

    policy, noise, pre, env_pre, mean, std = load_pi05()
    np.savez(f"{args.out}/norm.npz", mean=mean, std=std)
    for kind in args.dump:
        os.makedirs(f"{args.out}/{kind}", exist_ok=True)
        rows = [r for r in map(json.loads, open(f"{V}/pairs/{kind}.jsonl"))
                if os.path.exists(f"{V}/frames/{kind}/{r['key']}_a0.npz")
                and not os.path.exists(f"{args.out}/{kind}/{r['key']}_a0.npy")]
        print(f"[dump] {kind}: {len(rows)} episodes to do", flush=True)
        t0, pend = time.time(), []  # pend: (episode file, frames, row) waiting for a batch

        def flush(items):
            obs = preprocess_observation(raw_obs([(fr, i) for _, fr, i, _ in items]))
            obs["task"] = [lg for *_, lg in items]
            obs = pre(env_pre(obs))
            noise.set([zlib.crc32(f"{f}/{i}".encode()) for f, _, i, _ in items])
            with torch.inference_mode():
                return policy.predict_action_chunk(obs).float().cpu().numpy()

        for n_ep, r in enumerate(rows):
            fr = dict(np.load(f"{V}/frames/{kind}/{r['key']}_a0.npz"))
            out = np.full((len(fr["t"]), CHUNK, 7), np.nan, np.float32)
            calls = [i for i, t in enumerate(fr["t"]) if int(t) % CALL == 0]
            for b0 in range(0, len(calls), args.batch):
                items = [(r["key"], fr, i, lang[r["task_id"]]) for i in calls[b0:b0 + args.batch]]
                out[calls[b0:b0 + args.batch]] = flush(items)
            np.save(f"{args.out}/{kind}/{r['key']}_a0.npy", out.astype(np.float16))
            if n_ep % 50 == 0:
                print(f"[dump] {kind} {n_ep + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {args.out}", flush=True)


def refvis(args, preprocess_observation):
    """--refvis: the references' pi0.5 image tokens (V9 v2-encoder format) and preprocessed states (resumable)."""
    from pace.pi05 import batch_state, vis_tokens

    policy, _, pre, env_pre, _, _ = load_pi05()
    rows = [r for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))
            if r["success"] and os.path.exists(f"{V}/frames/refs/t{r['task_id']}_i{r['init']}_a0.npz")]
    t0 = time.time()
    for n, r in enumerate(rows):
        key = f"t{r['task_id']}_i{r['init']}"
        if os.path.exists(f"{args.out}/{key}_a0.npz"):
            continue
        fr = dict(np.load(f"{V}/frames/refs/{key}_a0.npz"))
        vis, st = [], []
        for b0 in range(0, len(fr["t"]), args.batch):
            items = [(fr, i) for i in range(b0, min(len(fr["t"]), b0 + args.batch))]
            obs = preprocess_observation(raw_obs(items))
            obs["task"] = [r["language"]] * len(items)
            obs = pre(env_pre(obs))
            with torch.inference_mode():
                images, img_masks = policy._preprocess_images(obs)
                embs = policy.model._embed_images(images, img_masks)
                vis.append(vis_tokens(embs, (0, 1)).half().cpu().numpy())
                st.append(batch_state(obs).cpu().numpy())
        np.savez(f"{args.out}/{key}_a0.npz", vis=np.concatenate(vis), state=np.concatenate(st), t=fr["t"])
        if n % 50 == 0:
            print(f"[refvis] {n + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"-> {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_expert", type=int, default=240, help="expert pair episodes (all their calls)")
    ap.add_argument("--n_aug", type=int, default=60, help="pi05_aug episodes for the input check")
    ap.add_argument("--batch", type=int, default=16, help="samples per forward pass (x 2 noise seeds)")
    ap.add_argument("--out", default="/root/pace/runs/v11/check")
    ap.add_argument("--dry", action="store_true", help="build the samples and the first batch's inputs only (no GPU)")
    ap.add_argument("--dump", nargs="+", default=None, help="pair kinds: save pi0.5's chunk at every call (V11 data)")
    ap.add_argument("--refvis", action="store_true", help="the references' pi0.5 image tokens and states (V11, V9 branch)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from lerobot.envs.utils import preprocess_observation

    lang = {r["task_id"]: r["language"] for r in map(json.loads, open(f"{V}/refs/episodes.jsonl"))}
    if args.dump:
        return dump(args, lang, preprocess_observation)
    if args.refvis:
        return refvis(args, preprocess_observation)
    rng = np.random.default_rng(0)

    # samples: (kind, frames npz, row, a* or recorded chunk (env units), meta)
    samples = []
    ex = [r for r in map(json.loads, open(f"{V}/pairs/expert.jsonl"))
          if r["success"] and os.path.exists(f"{V}/frames/expert/{r['key']}_a0.npz")]
    for r in [ex[i] for i in sorted(rng.choice(len(ex), min(args.n_expert, len(ex)), replace=False))]:
        fr = dict(np.load(f"{V}/frames/expert/{r['key']}_a0.npz"))
        pz = np.load(f"{V}/pairs/expert/{r['key']}.npz")
        acts, d = pz["actions"].astype(np.float64), pz["d"]
        aug = json.loads(str(fr["aug"]))
        for i, t in enumerate(fr["t"]):
            t = int(t)
            if t % CALL or t >= len(acts):
                continue
            ph = next((j for j, (s, e, *_) in enumerate(r["phases"]) if t < e), len(r["phases"]) - 1)
            samples.append(dict(kind="expert", fr=fr, row=i, target=target_chunk(acts, t), lang=lang[r["task_id"]],
                                key=r["key"], t=t, phase=ph, d=float(np.linalg.norm(d[min(t, len(d) - 1)][:2]) * 100),
                                cam_pos=float(np.linalg.norm(aug["cam_dpos"])), cam_angle=float(aug["cam_angle"])))
    pa = [r for r in map(json.loads, open(f"{V}/pi05_aug/episodes.jsonl"))
          if os.path.exists(f"{V}/frames/pi05_aug/t{r['task_id']}_i{r['init']}_a{r['aug_idx']}_a0.npz")]
    for r in [pa[i] for i in sorted(rng.choice(len(pa), min(args.n_aug, len(pa)), replace=False))]:
        key = f"t{r['task_id']}_i{r['init']}_a{r['aug_idx']}"
        fr = dict(np.load(f"{V}/frames/pi05_aug/{key}_a0.npz"))
        acts = np.load(f"{V}/pi05_aug/{r['actions']}").astype(np.float64)
        for i, t in enumerate(fr["t"]):
            t = int(t)
            if t % CALL or t >= len(acts):
                continue
            samples.append(dict(kind="pi05_aug", fr=fr, row=i, target=target_chunk(acts, t), lang=r["language"],
                                key=key, t=t, phase=-1, d=np.nan, success=bool(r["success"]),
                                cam_pos=float(np.linalg.norm(r["aug"]["cam_dpos"])), cam_angle=float(r["aug"]["cam_angle"])))
    print(f"[v11] {len(samples)} samples: expert {sum(s['kind'] == 'expert' for s in samples)}, "
          f"pi05_aug {sum(s['kind'] == 'pi05_aug' for s in samples)}", flush=True)
    if args.dry:
        o = preprocess_observation(raw_obs([(s["fr"], s["row"]) for s in samples[:4]]))
        print("[v11] dry run, first inputs:", {k: (tuple(v.shape), v.dtype) for k, v in o.items() if hasattr(v, "shape")},
              "| robot_state eef pos", tuple(o["observation.robot_state"]["eef"]["pos"].shape),
              "| target chunk", samples[0]["target"].shape, "| meta", {k: samples[0][k] for k in ("key", "t", "phase", "d", "cam_angle")})
        return

    policy, noise, pre, env_pre, mean, std = load_pi05()
    norm = lambda a: (a - mean) / std

    a1 = np.zeros((len(samples), CHUNK, 7), np.float32)
    a2 = np.zeros_like(a1)
    t0 = time.time()
    for b0 in range(0, len(samples), args.batch):
        sb = samples[b0:b0 + args.batch]
        rows = [(s["fr"], s["row"]) for s in sb] * 2  # the same inputs twice: noise seed 1, seed 2
        obs = preprocess_observation(raw_obs(rows))
        obs["task"] = [s["lang"] for s in sb] * 2
        obs = pre(env_pre(obs))
        noise.set([1_000_003 * (b0 + i) + 1 for i in range(len(sb))] + [1_000_003 * (b0 + i) + 2 for i in range(len(sb))])
        with torch.inference_mode():
            a = policy.predict_action_chunk(obs).float().cpu().numpy()
        a1[b0:b0 + len(sb)], a2[b0:b0 + len(sb)] = a[:len(sb)], a[len(sb):]
        if (b0 // args.batch) % 20 == 0:
            print(f"[v11] {b0 + len(sb)}/{len(samples)} ({time.time() - t0:.0f}s)", flush=True)

    tgt = np.stack([norm(s["target"]) for s in samples]).astype(np.float32)
    res, spr = dist(tgt, a1), dist(a2, a1)
    meta = {k: np.array([s.get(k, np.nan) for s in samples]) for k in ("kind", "key", "t", "phase", "d", "cam_pos", "cam_angle")}
    np.savez(f"{args.out}/check.npz", a_pi=a1.astype(np.float16), a_pi2=a2.astype(np.float16), target=tgt.astype(np.float16),
             **{f"res_{k}": v for k, v in res.items()}, **{f"spread_{k}": v for k, v in spr.items()}, **meta)

    lines = []
    for kind in ("expert", "pi05_aug"):
        m = meta["kind"] == kind
        lab = "a* - a_pi" if kind == "expert" else "recorded - offline a_pi"
        for k in ("rms_exec", "pos_exec", "rms_all"):
            lines.append(f"{kind:8s} {k:8s}: {lab} mean {res[k][m].mean():.3f} median {np.median(res[k][m]):.3f} | "
                         f"spread (two seeds) mean {spr[k][m].mean():.3f} median {np.median(spr[k][m]):.3f} | "
                         f"ratio {res[k][m].mean() / spr[k][m].mean():.2f}")
    m = meta["kind"] == "expert"
    for name, v, edges in (("cam_angle deg", meta["cam_angle"], (0, 4, 8, 12.1)), ("cam_pos m", meta["cam_pos"], (0, 0.06, 0.1, 0.2)),
                           ("|d| cm", meta["d"], (0, 1, 2, 4, 100)), ("phase", meta["phase"], (0, 1, 2, 3, 10))):
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = m & (v >= lo) & (v < hi)
            if g.sum():
                lines.append(f"  expert by {name} [{lo}, {hi}): n {g.sum():5d}  residual rms_exec {res['rms_exec'][g].mean():.3f}"
                             f"  spread {spr['rms_exec'][g].mean():.3f}  ratio {res['rms_exec'][g].mean() / spr['rms_exec'][g].mean():.2f}")
    open(f"{args.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
