"""
V10-1b closed loop on the generalization test (10-24; docs/PACE_V1-V8.md "V10-1b 的判定标准"). The user chose to run it
although run1 missed the d gate by 0.01 cm (1.01 cm at < 3 cm; registered <= 1 cm): a deliberate deviation.
Cells: the 20 camera-viewpoint variants x init states 1..5 (runs/rectest/v10_shift/base_targets), env seed
1000 + 7 * task + init, references = the same 3 sources per cell as the true-d upper bound (v10_oracle_noise.py:
pi0.5 successes on light / noise / background variants of the same base task, init 0, runs/v10b/noise_sources).
Everything runs through lerobot's own rollout (pi0.5 called every step exactly as in its evaluation); the method sits
in an env wrapper that receives pi0.5's action and may replace it:
  base : never replaces (pi0.5 alone; two seeds of pi0.5's sampling, 1000 and 5000, on this GPU model);
  v10  : at the first call renders both cameras at 224 (as the training frames), runs the frozen SigLIP + PCA and the
         comparator against the reference's frames at the aligned steps (the timeline: j = t) and the phase end;
         follows the reference for the whole episode iff P(follow succeeds) - P(pi0.5 succeeds) > --margin (0, set on
         validation: damage 3.9%); while following, every call re-estimates d and the action is
         a_ref[t] + clip(((p_ref[t] + d) - p) / 0.05) (the true-d experiment's controller, d from the network); after
         the reference's last step it holds 40 steps, then hands back to pi0.5. Not following = pi0.5's action
         unchanged (the episode is bit-identical to base with the same seed).
    python closed_loop.py --check          # the 224 render used for the network == a 224 env's observation
    python closed_loop.py --modes v10 base --out runs/v10b/closed/run1
10-24 (after run1): pi0.5's noise is drawn per episode (--noise episode, RowNoise) and the wrapper log is written at
every call, so failed episodes keep their decision; --check_noise runs 8 base episodes in two batch orders.
"""
import argparse
import functools
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

import collect_pairs as P
import render_pairs as R
import v10_oracle_shift2 as O

ROOT = "/root/pace/runs"
CALL, HOLD = 10, 40
_CACHE = {}


def render224(inner):
    """Both cameras at 224 x 224 from the current sim state; equal to a 224 env's observation images, i.e. to the
    training frames (--check 10-24: unflipped mean abs diff 0.000 for both cameras)."""
    return {c: np.ascontiguousarray(inner.sim.render(camera_name=c, width=224, height=224))
            for c in ("agentview", "robot0_eye_in_hand")}


def _models(cfg):
    if "m" not in _CACHE:
        import train_v10b as T

        ck = torch.load(cfg["ckpt"], weights_only=False)
        a = ck["args"]
        z = np.load(cfg["pca"])
        model = T.Comparator(z["comp"].shape[0], w=a["width"], pool=a["pool"]).cuda().eval()
        model.load_state_dict(ck["model"])
        tower = torch.load(cfg["tower"], weights_only=False).cuda().eval()
        _CACHE["m"] = (model, tower, torch.from_numpy(z["mean"]).cuda(), torch.from_numpy(z["comp"]).cuda(), a["d_scale"])
    return _CACHE["m"]


@torch.no_grad()
def encode(tower, mean, comp, imgs):
    x = torch.from_numpy(np.stack(imgs)).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1.0
    h = tower(pixel_values=x.to(next(tower.parameters()).dtype)).last_hidden_state.float()
    return (h - mean) @ comp.T  # [k, 256, D]


class V10Env(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg, self.written = cfg, False  # the log is written once: lerobot autoresets finished envs

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.t, self.follow, self.d, self.log = 0, None, np.zeros(2), []
        return obs, info

    @torch.no_grad()
    def _infer(self, inner):
        c, src = self.cfg, self.cfg["src"]
        model, tower, mean, comp, d_scale = _models(c)
        im = render224(inner)
        cur = encode(tower, mean, comp, [im["agentview"], im["robot0_eye_in_hand"]])[None]  # [1, 2, 256, D]
        t = self.t
        ph = next((i for i, (s, e) in enumerate(src["phases"]) if t < max(e, s + 1)), len(src["phases"]) - 1)
        s, e = src["phases"][ph]
        rt = src["t"]
        rows = [int(np.argmin(np.abs(rt - w))) for w in (t - CALL, t, t + CALL, t + 2 * CALL, e)]
        ref = torch.from_numpy(src["feats"][rows].astype(np.float32)).cuda()
        feats = torch.cat([cur, ref])[None]  # [1, 6, 2, 256, D]
        robot = inner.robots[0]
        eef = np.array(inner.sim.data.site_xpos[robot.eef_site_id])
        grip = np.array(inner.sim.data.qpos[robot._ref_gripper_joint_pos_indexes])
        j = rows[1]
        state = np.r_[eef - src["eef"][j], grip, src["grip"][j], min((t - s) / max(e - s, 1), 1.5), t / 600.0]
        st = torch.tensor(state, dtype=torch.float32).cuda()[None]
        lang = torch.tensor(np.stack(c["lang"]), dtype=torch.float32).cuda()[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, lf, lb = model(feats, st, lang, torch.zeros(1, dtype=torch.bool).cuda(), d_scale)
        return d.float()[0].cpu().numpy(), float(torch.sigmoid(lf.float())[0]), float(torch.sigmoid(lb.float())[0])

    def step(self, action):
        a = np.asarray(action, dtype=np.float64).copy()
        c = self.cfg
        if c["mode"] == "v10":
            src = c["src"]
            n = len(src["actions"])
            inner = self.env.unwrapped._env.env
            if self.t % CALL == 0 and self.t < n and self.follow is not False:
                d, pf, pb = self._infer(inner)
                if self.follow is None:
                    self.follow = (pf - pb) > c["margin"]
                self.d = d
                self.log.append(dict(t=self.t, d=[round(float(x), 4) for x in d], pf=round(pf, 3), pb=round(pb, 3)))
                self._dump(None)  # also failed episodes keep their decision (run1 logged only on success / termination)
            if self.follow and self.t < n + HOLD:
                tt = min(self.t, n - 1)
                ar = src["actions"][tt].copy() if self.t < n else np.r_[np.zeros(6), src["actions"][-1][6]]
                hand = np.array(inner.sim.data.site_xpos[inner.robots[0].eef_site_id])
                tgt = src["path"][tt] + np.r_[self.d, 0.0]
                ar[:3] = np.clip(ar[:3] + (tgt - hand) / 0.05, -1.0, 1.0)
                a = ar
        obs, r, term, trunc, info = self.env.step(a)
        self.t += 1
        if term or trunc or info.get("is_success"):
            self._dump(bool(info.get("is_success")))
            self.written = True
        return obs, r, term, trunc, info

    def _dump(self, success):
        if self.cfg["mode"] == "v10" and self.cfg.get("log") and not self.written:
            with open(self.cfg["log"], "w") as f:
                json.dump(dict(follow=bool(self.follow), calls=self.log, steps=self.t, success=success), f)


class RowNoise:
    """pi0.5's flow-matching noise drawn per episode (row) from a generator seeded by that episode's seed, instead of
    one draw for the whole batch from the global RNG: an episode's noise then no longer depends on which episodes
    share its batch, so a V10-1b episode that does not follow sees exactly the noise of the base episode with the
    same seed (run1 shared the draw across the batch; 10-24)."""

    def __init__(self):
        self.gens = None

    def set(self, seeds):
        self.gens = [torch.Generator().manual_seed(int(s)) for s in seeds]

    def __call__(self, shape, device):
        assert self.gens is not None and shape[0] == len(self.gens), (shape, len(self.gens or []))
        return torch.stack([torch.randn(tuple(shape[1:]), generator=g) for g in self.gens]).to(device)


def v10_env(func, k, cfg, **kw):
    return V10Env(func(k, **kw), cfg)


def prepare_refs(srcs, out, tower, mean, comp):
    """Each source replayed in its own scene at 224: per-call features (fp16) and state, cached."""
    os.makedirs(f"{out}/refs", exist_ok=True)
    for s in srcs:
        path = f"{out}/refs/t{s['task_id']}_i{s['init_index']}.npz"
        if not os.path.exists(path):
            ok, _, rec = R.replay(s["task_id"], s["init_index"], s["seed"], s["actions"], None)
            assert ok, f"source {s['key']} does not replay"
            f = torch.stack([encode(tower, mean, comp, list(rec[c])) for c in ("agent", "wrist")], 1)
            np.savez(path, feats=f.half().cpu().numpy(), t=rec["t"], eef=rec["eef_pos"], grip=rec["gripper"])
        z = np.load(path)
        s.update(feats=z["feats"], t=z["t"], eef=z["eef"], grip=z["grip"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/v10b/train/run1/best.pt")
    ap.add_argument("--pca", default=f"{ROOT}/v10b/feats/pca.npz")
    ap.add_argument("--out", default=f"{ROOT}/v10b/closed/run1")
    ap.add_argument("--modes", nargs="+", default=["v10", "base"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--noise", choices=["episode", "batch"], default="episode")
    ap.add_argument("--check_noise", action="store_true", help="base on 8 cells in two batch orders: identical?")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/wrapper", exist_ok=True)
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _LazyAsyncVectorEnv
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from pace.common import load_policy
    from pace.eval_suite import FixedTasks
    from pace.pi05 import InstructionText

    P.worker_init()
    G = O.G
    policy, _ = load_policy("lerobot/pi05_libero_finetuned", "cuda", n_action_steps=10)
    noise = RowNoise()
    if args.noise == "episode":
        policy.model.sample_noise = noise
    tower_path = f"{ROOT}/v10b/closed/siglip_tower.pt"
    if not os.path.exists(tower_path):
        torch.save(policy.model.paligemma_with_expert.paligemma.model.vision_tower, tower_path)
    z = np.load(args.pca)
    tower = torch.load(tower_path, weights_only=False).cuda().eval()
    mean, comp = torch.from_numpy(z["mean"]).cuda(), torch.from_numpy(z["comp"]).cuda()
    targets = [json.loads(l) for l in open(f"{ROOT}/rectest/v10_shift/base_targets/episodes.jsonl")]
    cams = sorted({r["task_id"]: dict(r, basename=r["name"].split("_view_")[0]) for r in targets}.items())

    if args.check:
        tid, r = cams[0]
        env_cfg = LiberoPlusEnv(task="libero_10")
        gk = dict(env_cfg.gym_kwargs)
        gk.pop("task_ids", None)
        fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=G["cams"],
                           episode_length=env_cfg.episode_length, init_states=env_cfg.init_states, gym_kwargs=gk,
                           control_mode=env_cfg.control_mode, camera_name_mapping=env_cfg.camera_name_mapping,
                           is_libero_plus=env_cfg.is_libero_plus)[0]
        e1 = fn.func(1, **fn.keywords)
        e1.reset(seed=1)
        a = render224(e1._env.env)
        e2 = R.make_env(tid, 1, 224)
        e2.reset(seed=1)
        o = e2._env.env._get_observations(force_update=True)
        for c in a:
            b = o[c + "_image"]
            print(f"[check] {c}: identical {np.array_equal(a[c], b)}, mean abs diff {np.abs(a[c].astype(int) - b).mean():.3f}"
                  f"; vs unflipped {np.abs(a[c][::-1].astype(int) - b).mean():.3f}", flush=True)
        return

    rows = [json.loads(l) for l in open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")]
    sources = {}
    for r in rows:
        if r["success"]:
            sources[(r["base"], r["task_id"])] = dict(key=(r["base"], r["task_id"]), task_id=int(r["task_id"]),
                                                      init_index=int(r["init"]), seed=int(r["seed"]),
                                                      language=r["language"],
                                                      actions=np.load(f"{ROOT}/v10b/noise_sources/{r['actions']}").astype(np.float64))
    for s in sources.values():
        rec = O.record_source(s)
        s["rec_ok"] = rec["success"]
        s["phases"] = [(int(a), int(b)) for a, b, _ in O.phase_targets(rec)]
        n = len(s["actions"])
        s["path"] = np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])
    prepare_refs([s for s in sources.values() if s["rec_ok"]], args.out, tower, mean, comp)
    del tower
    torch.cuda.empty_cache()
    idx_lang = [json.loads(l)["language"] for l in open(f"{ROOT}/v10b/refs/episodes.jsonl")]
    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(dict.fromkeys(idx_lang)))

    entries = []
    for tid, r in cams:
        pool = [s for (b, _), s in sources.items() if b == r["basename"] and s["rec_ok"]][:3]
        for k in (1, 2, 3, 4, 5):
            seed = 1000 + 7 * tid + k
            if "v10" in args.modes:
                for s in pool:
                    entries.append(dict(mode="v10", task_id=tid, init=k, seed=seed, language=r["base"], src=s,
                                        key=f"v10_t{tid}_i{k}_s{s['task_id']}"))
            if "base" in args.modes:
                for sd in (seed, seed + 4000):
                    entries.append(dict(mode="base", task_id=tid, init=k, seed=sd, language=r["base"], src=None,
                                        key=f"base_t{tid}_i{k}_seed{sd}"))
    if args.check_noise:  # the same 8 base episodes in two batch orders
        base = [e for e in entries if e["mode"] == "base"][:16:2]
        entries = [dict(e, key=e["key"] + "_o1") for e in base] + [dict(e, key=e["key"] + "_o2") for e in base[::-1]]
    log_path = f"{args.out}/episodes.jsonl"
    done = {json.loads(l)["key"] for l in open(log_path)} if os.path.exists(log_path) else set()
    todo = [e for e in entries if e["key"] not in done]
    print(f"[closed] {len(entries)} episodes ({len(todo)} to do): "
          + ", ".join(f"{m} {sum(e['mode'] == m for e in entries)}" for m in args.modes), flush=True)

    env_cfg = LiberoPlusEnv(task="libero_10")
    pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                         preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                                 "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    log = open(log_path, "a")
    t0 = time.time()
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        fns = []
        for e in batch:
            fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=e["task_id"], n_envs=1,
                               camera_names=G["cams"], episode_length=env_cfg.episode_length,
                               init_states=env_cfg.init_states, gym_kwargs=gk, control_mode=env_cfg.control_mode,
                               camera_name_mapping=env_cfg.camera_name_mapping, is_libero_plus=env_cfg.is_libero_plus)[0]
            cfg = dict(mode=e["mode"], margin=args.margin, ckpt=args.ckpt, pca=args.pca, tower=tower_path,
                       log=f"{args.out}/wrapper/{e['key']}.json")
            if e["src"] is not None:
                s = e["src"]
                cfg["src"] = {k: s[k] for k in ("actions", "path", "phases", "feats", "t", "eef", "grip")}
                cfg["lang"] = [text.embed(e["language"]).numpy(), text.embed(s["language"]).numpy()]
            fns.append(functools.partial(v10_env, fn.func, e["init"], cfg, **fn.keywords))
        noise.set([e["seed"] for e in batch])
        env = _LazyAsyncVectorEnv(fns)
        try:
            torch.manual_seed(batch[0]["seed"])
            ret = rollout(env=env, policy=policy, env_preprocessor=FixedTasks(env_pre, [e["language"] for e in batch]),
                          env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                          seeds=[e["seed"] for e in batch])
        finally:
            env.close()
        succ = ret["success"].bool().any(dim=1).tolist()
        dn = ret["done"].bool()
        for i, e in enumerate(batch):
            T = int(torch.argmax(dn[i].int())) + 1 if bool(dn[i].any()) else dn.shape[1]
            w = f"{args.out}/wrapper/{e['key']}.json"
            extra = json.load(open(w)) if os.path.exists(w) else {}
            log.write(json.dumps(dict(key=e["key"], mode=e["mode"], task_id=e["task_id"], init=e["init"], seed=e["seed"],
                                      source=e["src"]["task_id"] if e["src"] else None, success=bool(succ[i]), steps=T,
                                      follow=extra.get("follow"), gpu=torch.cuda.get_device_name(0))) + "\n")
        log.flush()
        print(f"[closed] {b0 + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s), batch success {sum(succ)}/{len(batch)}",
              flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
