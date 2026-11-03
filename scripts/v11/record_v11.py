"""
Side-by-side videos for V11 (10-30; user: "在 3 个纠正的样例上录制一下原本的和我们这个的"). Re-runs chosen episodes of
the closed-loop test (closed_loop_v11.py, runs/v11/closed/run1) with the same env seed, pi0.5 noise seed, reference
and checkpoint, records lerobot's render frame (env.call("render")) at every step, and writes one mp4 per example:
left = pi0.5 alone (base), right = pi0.5 + V11. The re-run outcome is printed next to the logged one (reproducibility).
    python record_v11.py --pairs 840:2:2433 934:4:2468 1098:2:2338 --out runs/v11/closed/videos
10-31 (harm diagnosis): --log may name several runs; non-camera variants are found in noncam_targets.json; every V11 call
stores how far the arm's real progress is from the time-aligned reference step: the reference call whose state (hand
position, gripper; pi0.5's normalized state) is nearest to the current one vs the call time alignment reads (trace.json).
"""
import argparse
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

import closed_loop_v11 as C
import v10_oracle_shift2 as O

ROOT = C.ROOT


def label(img, lines, color):
    im = np.ascontiguousarray(img).copy()
    h = 26 * len(lines) + 8
    im[:h] = (im[:h] * 0.35).astype(np.uint8)
    for i, (s, c) in enumerate(zip(lines, color)):
        cv2.putText(im, s, (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="task:init:source (the v11 episode; base = same seed)")
    ap.add_argument("--ckpt", default=f"{ROOT}/v11/train/v9_run2_4090/best.pt")
    ap.add_argument("--refs", default=f"{ROOT}/v11/closed/run1", help="where closed_loop_v11.py cached the references")
    ap.add_argument("--log", nargs="+", default=[f"{ROOT}/v11/closed/run1/episodes.jsonl"])
    ap.add_argument("--out", default=f"{ROOT}/v11/closed/videos")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _LazyAsyncVectorEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from pace.eval_suite import FixedTasks
    from v11_check import load_pi05

    O.worker_init()
    policy, noise, pre, env_pre, _, _ = load_pi05()
    _, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                       preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                               "rename_observations_processor": {"rename_map": {}}})
    _, env_post = make_env_pre_post_processors(O.G["env_cfg"], policy.config)
    v11 = C.V11(policy, args.ckpt)
    logged = {json.loads(l)["key"]: json.loads(l) for f in args.log for l in open(f)}
    targets = {json.loads(l)["task_id"]: json.loads(l) for l in open(f"{ROOT}/rectest/v10_shift/base_targets/episodes.jsonl")}
    for x in json.load(open(f"{ROOT}/rectest/v10_shift/noncam_targets.json")):  # pi0.5 gets the variant's own instruction
        targets[x["task_id"]] = dict(x, base=x["language"])
    trace, orig_state = {}, v11.batch_state
    D = [0, 1, 2, 6, 7]  # hand position, gripper

    def state_tap(obs):
        st = orig_state(obs)
        if v11.rows is not None:
            t = v11.calls * C.CALL
            cur = st.detach().float().cpu().numpy()
            for i, r in enumerate(v11.rows):
                dist = np.linalg.norm(r["state"][:, D] - cur[i, D], axis=1)
                jt, jn = int(np.argmin(np.abs(r["t"] - t))), int(np.argmin(dist))
                trace.setdefault(i, []).append([t, int(r["t"][jt]), int(r["t"][jn]), float(dist[jt]), float(dist[jn])])
        return st

    v11.batch_state = state_tap
    sources = {}
    for r in map(json.loads, open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")):
        if r["success"]:
            sources[r["task_id"]] = dict(task_id=int(r["task_id"]), init_index=int(r["init"]), seed=int(r["seed"]),
                                         base=r["base"], language=r["language"],
                                         actions=np.load(f"{ROOT}/v10b/noise_sources/{r['actions']}").astype(np.float64))
    ex = []
    for p in args.pairs:
        tid, k, src = map(int, p.split(":"))
        ex.append(dict(task_id=tid, init=k, seed=1000 + 7 * tid + k, src=sources[src],
                       language=targets[tid]["base"], name=targets[tid]["name"]))  # pi0.5's instruction as in the test
    refs = {e["src"]["task_id"]: C.prepare_ref(e["src"], args.refs, policy, pre, env_pre, preprocess_observation) for e in ex}

    runs = {}
    for mode in ("base", "v11"):
        frames = [[] for _ in ex]

        def grab(env):
            for i, f in enumerate(env.call("render")):
                frames[i].append(np.asarray(f))

        fns = []
        for e in ex:
            fn = C.make_env(e["task_id"], e["init"])
            fns.append(C.functools.partial(fn.func, e["init"], **fn.keywords))
        v11.set_rows(None if mode == "base" else [refs[e["src"]["task_id"]] for e in ex])
        noise.set([e["seed"] for e in ex])
        env = _LazyAsyncVectorEnv(fns)
        try:
            torch.manual_seed(ex[0]["seed"])
            ret = rollout(env=env, policy=policy, env_preprocessor=FixedTasks(env_pre, [e["language"] for e in ex]),
                          env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                          seeds=[e["seed"] for e in ex], render_callback=grab)
        finally:
            env.close()
        succ = ret["success"].bool().any(dim=1).tolist()
        dn = ret["done"].bool()
        T = [int(torch.argmax(dn[i].int())) + 1 if bool(dn[i].any()) else dn.shape[1] for i in range(len(ex))]
        runs[mode] = dict(frames=frames, succ=succ, T=T, delta=None if mode == "base" else v11.stats)
    v11.set_rows(None)

    rows = []
    for i, e in enumerate(ex):
        tr = np.array(trace[i])[:(runs["v11"]["T"][i] + C.CALL - 1) // C.CALL]
        off = np.abs(tr[:, 2] - tr[:, 1])  # steps between the nearest and the time-aligned reference call
        rows.append(dict(task=e["task_id"], init=e["init"], source=e["src"]["task_id"], success=runs["v11"]["succ"][i],
                         steps=runs["v11"]["T"][i], base_success=runs["base"]["succ"][i], base_steps=runs["base"]["T"][i],
                         offset_median=float(np.median(off)), offset_p90=float(np.percentile(off, 90)),
                         dist_time=float(np.median(tr[:, 3])), dist_near=float(np.median(tr[:, 4])),
                         delta=float(np.mean(runs["v11"]["delta"][i]["delta"][:len(tr)])), trace=tr.tolist()))
        print(f"[align] t{e['task_id']} i{e['init']} s{e['src']['task_id']} v11 {rows[-1]['success']} ({rows[-1]['steps']}), "
              f"base {rows[-1]['base_success']}: |offset| median {rows[-1]['offset_median']:.0f} p90 "
              f"{rows[-1]['offset_p90']:.0f} steps, state dist time {rows[-1]['dist_time']:.3f} nearest {rows[-1]['dist_near']:.3f}",
              flush=True)
    json.dump(rows, open(f"{args.out}/trace.json", "w"))
    for i, e in enumerate(ex):
        kb = f"base_t{e['task_id']}_i{e['init']}_seed{e['seed']}"
        kv = f"v11_t{e['task_id']}_i{e['init']}_s{e['src']['task_id']}"
        print(f"[rec] {e['name']} init {e['init']}: base rerun {runs['base']['succ'][i]} ({runs['base']['T'][i]} steps), "
              f"logged {logged.get(kb, {}).get('success')} ({logged.get(kb, {}).get('steps')}); v11 rerun "
              f"{runs['v11']['succ'][i]} ({runs['v11']['T'][i]} steps), logged {logged.get(kv, {}).get('success')} "
              f"({logged.get(kv, {}).get('steps')})", flush=True)
        n = max(runs["base"]["T"][i], runs["v11"]["T"][i]) + 1  # frame 0 = after reset
        path = f"{args.out}/v11_t{e['task_id']}_i{e['init']}_s{e['src']['task_id']}.mp4"
        w = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
        green, red, white = (90, 230, 90), (90, 90, 240), (255, 255, 255)
        for t in range(n):
            tiles = []
            for mode, title in (("base", "pi0.5 (base)"), ("v11", "pi0.5 + V11")):
                r = runs[mode]
                fr = r["frames"][i][min(t, r["T"][i])]
                over = t >= r["T"][i]
                status = ("SUCCESS" if r["succ"][i] else "FAILED") if over else f"step {t}"
                col = (green if r["succ"][i] else red) if over else white
                tiles.append(label(fr, [title, status], [white, col]))
            img = np.concatenate([tiles[0], np.full((tiles[0].shape[0], 6, 3), 255, np.uint8), tiles[1]], 1)
            w.append_data(img)
        w.close()
        print(f"[rec] -> {path} ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
