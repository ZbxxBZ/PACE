"""
Why does V11 rescue LIVING_ROOM_SCENE5 (mugs on plates) when another task's reference does not (11-01, user: "可以先看看吧";
desktop design doc 10.14f, tilt check: base 0/9, v11 8/9, other 2/9)? Re-runs the 9 cells of tilt variants 838, 842, 846
with the seeds of runs/v11/closed/tilt, three rollouts per cell: base, v11 (own reference), other (other-task reference, the
logged one). Whatever is executed, at every pi0.5 call Delta is also computed under each reference condition:
  true / other / same2 (another successful reference of the same task),
  act_true / vis_true / state_true (other-task reference with only that part from the true one: the action chunk; the image
  tokens; the state with the position scalars).
Stored per call: the first 10 steps of each Delta (commanded hand motion in cm, action * 5 cm, OSC scale 0.05 m), its gripper
part and rms, the same for a_pi, the normalized hand state. mp4 and a contact sheet per cell (rows base / v11 / other).
    python lr5_probe.py --out runs/v11/closed/lr5
"""
import argparse
import collections
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

import closed_loop_v11 as C
import v10_oracle_shift2 as O

ROOT = C.ROOT
CONDS = dict(true=("true", "true", "true"), other=("other", "other", "other"), same2=("same2", "same2", "same2"),
             act_true=("true", "other", "other"), vis_true=("other", "true", "other"), state_true=("other", "other", "true"))
TILES = (("base", "pi0.5 (base)"), ("v11", "V11, own reference"), ("other", "V11, other-task reference"))


def tracked(fn, *a, **kw):
    """The env with obj_pos(): world position of every task object, read in the env's own process."""
    env = fn(*a, **kw)

    def obj_pos():
        e = env._env.env
        return {n: e.sim.data.body_xpos[e.sim.model.body_name2id(o.root_body)].round(4).tolist() for n, o in e.objects_dict.items()}

    env.obj_pos = obj_pos
    return env


def label(img, lines, colors):
    im = np.ascontiguousarray(img).copy()
    h = 22 * len(lines) + 6
    im[:h] = (im[:h] * 0.35).astype(np.uint8)
    for i, (s, c) in enumerate(zip(lines, colors)):
        cv2.putText(im, s, (6, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/v11/train/v9_run2_4090/best.pt")
    ap.add_argument("--refs", default=f"{ROOT}/v11/closed/run1")
    ap.add_argument("--out", default=f"{ROOT}/v11/closed/lr5")
    ap.add_argument("--tasks", nargs="+", type=int, default=[838, 842, 846])
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first cells only")
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

    targets = {x["task_id"]: x for x in json.load(open(f"{ROOT}/v11/closed/tilt/targets.json"))}
    sources = {}
    for r in map(json.loads, open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")):
        if r["success"]:
            sources[r["task_id"]] = dict(task_id=int(r["task_id"]), init_index=int(r["init"]), seed=int(r["seed"]),
                                         base=r["base"], language=r["language"],
                                         actions=np.load(f"{ROOT}/v10b/noise_sources/{r['actions']}").astype(np.float64))
    logged = collections.defaultdict(dict)
    for r in map(json.loads, open(f"{ROOT}/v11/closed/tilt/episodes.jsonl")):
        if r["task_id"] in args.tasks:
            logged[(r["task_id"], r["init"])][r["mode"]] = r
    refs = {}

    def ref(src):
        if src["task_id"] not in refs:
            refs[src["task_id"]] = C.prepare_ref(src, args.refs, policy, pre, env_pre, preprocess_observation)
        return refs[src["task_id"]]

    ex = []
    for (tid, k), m in sorted(logged.items()):
        tg = targets[tid]
        own = [s for s in sources.values() if s["base"] == tg["base"]][:3]
        t_src = sources[m["v11"]["source"]]
        s2 = own[(own.index(t_src) + 1) % len(own)]
        ex.append(dict(task_id=tid, init=k, seed=1000 + 7 * tid + k, language=tg["language"], src=t_src["task_id"],
                       src2=s2["task_id"], other=m["v11other"]["source"], rows=dict(true=ref(t_src), same2=ref(s2),
                                                                                  other=ref(sources[m["v11other"]["source"]])),
                       logged={x: m[x]["success"] for x in ("base", "v11", "v11other")}))
    if args.limit:
        ex = ex[:args.limit]
    print(f"[lr5] {len(ex)} cells, {len(refs)} references, conditions {list(CONDS)}", flush=True)

    S = dict(apply=None, rec=None)
    std, mean = v11.std[:7].astype(np.float32), v11.mean[:7].astype(np.float32)

    def read(r, t):
        j = int(np.argmin(np.abs(r["t"] - t)))
        L = len(r["t"])
        return dict(vis=torch.from_numpy(r["vis"][j].astype(np.float32)), state=torch.from_numpy(r["state"][j].astype(np.float32)),
                    act=torch.from_numpy(v11.norm(C.V.chunk(r["actions"], int(r["t"][j])))), pos=[j / max(L - 1, 1), j, L])

    def brief(x):
        """Normalized [10, 7] -> commanded hand motion over the 10 executed steps (cm), gripper mean (action units)."""
        d = x * std
        return dict(cm=[round(float(v), 3) for v in d[:, :3].sum(0) * 5], grip=round(float(d[:, 6].mean()), 4))

    @torch.no_grad()
    def probe(obs, a_pi, t):
        p, dev, B = policy, v11.pace.device, a_pi.shape[0]
        lm_cfg = p.model.paligemma_with_expert.paligemma.model.language_model.config
        attn = lm_cfg._attn_implementation  # prefix_cache switches it to eager: restore, pi0.5's own sampling unchanged
        try:
            images, img_masks = p._preprocess_images(obs)
            cur_vis = v11.vis_tokens(p.model._embed_images(images, img_masks), (0, 1)).float()
            cur_state = v11.batch_state(obs)
            prefix = v11.prefix_cache(p, obs)
        finally:
            lm_cfg._attn_implementation = attn
        x = F.pad(a_pi.float(), (0, p.config.max_action_dim - a_pi.shape[-1]))
        tm = torch.full((B,), v11.read_time, device=dev)

        def hidden():
            p.model.denoise_step(prefix_pad_masks=prefix[0], past_key_values=prefix[1], x_t=x, timestep=tm)
            return v11.cap.h.float()

        v11.pace.set_exp(None)
        h_off = hidden()
        out, gates = {}, {}
        for name, (ka, kv, ks) in CONDS.items():
            ra = [read(e["rows"][ka], t) for e in ex]
            rv = [read(e["rows"][kv], t) for e in ex]
            rs = [read(e["rows"][ks], t) for e in ex]
            a_ref = torch.stack([r["act"] for r in ra]).to(dev)
            ref_vis = torch.stack([r["vis"] for r in rv]).to(dev)
            ref_state = torch.stack([r["state"] for r in rs]).to(dev)
            rp = torch.tensor([r["pos"] for r in rs], dtype=torch.float32, device=dev)
            sim = (cur_vis * ref_vis).sum(-1).mean(-1)
            call = torch.full((B,), t / C.CALL, device=dev)
            scalars = torch.stack([rp[:, 0], sim, torch.zeros_like(sim), call / v11.call_scale, rp[:, 1] / v11.call_scale,
                                   rp[:, 2] / v11.call_scale], -1)[:, None]
            exp = dict(actions=a_ref[:, None], scalars=scalars, success=torch.ones(B, 1, dtype=torch.long, device=dev),
                       valid=torch.ones(B, 1, dtype=torch.bool, device=dev), ref_vis=ref_vis[:, None],
                       ref_state=ref_state[:, None], cur_vis=cur_vis, cur_state=cur_state)
            v11.adapter.relevance_stats = {}
            v11.pace.set_exp(exp)
            try:
                h_on = hidden()
                gates[name] = float(v11.adapter.relevance_stats.get("relevance_mean", float("nan")))
            finally:
                v11.pace.set_exp(None)
            out[name] = v11.head(h_on, h_off)[0]
        ap_np = a_pi[:, :C.EXEC, :7].float().cpu().numpy()
        for i in range(B):
            row = dict(t=t, state=[round(float(v), 4) for v in cur_state[i, [0, 1, 2, 6, 7]].cpu()], a_pi=brief(ap_np[i] + mean / std))
            for name, d in out.items():
                n = d[i, :C.EXEC].float().cpu().numpy()
                row[name] = dict(brief(n), rms=round(float(np.sqrt((n ** 2).mean())), 4))
            row["gate"] = {k: round(v, 3) for k, v in gates.items()}
            S["rec"][i].append(row)
        return out

    wrapped = policy.predict_action_chunk

    def pred(batch, **kw):
        act = wrapped(batch, **kw)  # pi0.5's chunk (the V11 wrapper has no rows); v11.calls is already incremented
        D = probe(batch, act, (v11.calls - 1) * C.CALL)
        return act if S["apply"] is None else act + D[S["apply"]].to(act.dtype)

    policy.predict_action_chunk = pred
    v11.set_rows(None)

    runs = {}
    for mode, apply in (("base", None), ("v11", "true"), ("other", "other")):
        frames = [[] for _ in ex]
        track = [[] for _ in ex]

        def grab(env):
            for i, f in enumerate(env.call("render")):
                frames[i].append(np.asarray(f))
            if (len(frames[0]) - 1) % 5 == 0:
                for i, p in enumerate(env.call("obj_pos")):
                    track[i].append(p)

        fns = []
        for e in ex:
            fn = C.make_env(e["task_id"], e["init"])
            fns.append(C.functools.partial(tracked, fn.func, e["init"], **fn.keywords))
        S.update(apply=apply, rec=[[] for _ in ex])
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
        runs[mode] = dict(frames=frames, succ=succ, T=T, rec=[S["rec"][i][:(T[i] + C.CALL - 1) // C.CALL] for i in range(len(ex))],
                          track=[track[i][:T[i] // 5 + 1] for i in range(len(ex))])
        print(f"[lr5] {mode}: {sum(succ)}/{len(ex)} success, steps {T}", flush=True)

    out = []
    for i, e in enumerate(ex):
        name = f"t{e['task_id']}_i{e['init']}"
        print(f"[lr5] {name} own s{e['src']} same2 s{e['src2']} other s{e['other']}: rerun base {runs['base']['succ'][i]} v11 "
              f"{runs['v11']['succ'][i]} other {runs['other']['succ'][i]}; logged {e['logged']}", flush=True)
        out.append(dict(task=e["task_id"], init=e["init"], own=e["src"], same2=e["src2"], other=e["other"], logged=e["logged"],
                        runs={m: dict(success=runs[m]["succ"][i], steps=runs[m]["T"][i], calls=runs[m]["rec"][i],
                                      track=runs[m]["track"][i]) for m in runs}))
        n = max(runs[m]["T"][i] for m in runs) + 1
        green, red, white = (90, 230, 90), (90, 90, 240), (255, 255, 255)

        def tile(m, title, t):
            r = runs[m]
            fr = r["frames"][i][min(t, r["T"][i])]
            over = t >= r["T"][i]
            status = ("SUCCESS" if r["succ"][i] else "FAILED") if over else f"step {t}"
            return label(fr, [title, status], [white, (green if r["succ"][i] else red) if over else white])

        w = imageio.get_writer(f"{args.out}/lr5_{name}.mp4", fps=20, codec="libx264", quality=7, macro_block_size=8)
        for t in range(n):
            ts = [tile(m, title, t) for m, title in TILES]
            gap = np.full((ts[0].shape[0], 6, 3), 255, np.uint8)
            w.append_data(np.concatenate([ts[0], gap, ts[1], gap, ts[2]], 1))
        w.close()
        cols = list(range(0, n, max(40, n // 9 + 1)))
        sheet = [np.concatenate([cv2.resize(tile(m, title, t), (240, 240)) for t in cols], 1) for m, title in TILES]
        imageio.imwrite(f"{args.out}/sheet_{name}.png", np.concatenate(sheet, 0))
    json.dump(out, open(f"{args.out}/probe.json", "w"))
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
