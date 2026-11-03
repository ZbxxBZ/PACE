"""
V11 closed loop (10-30; LeRobot pi05_libero_finetuned; desktop design doc 10.14e "训完后的计划" 2-3). The generalization
test of V10: the 20 camera-viewpoint variants x init states 1..5 (runs/rectest/v10_shift/base_targets), env seed
1000 + 7 * task + init, and the same references per cell as V10 (runs/v10b/noise_sources: pi0.5's successes on light /
noise / background variants of the same base task, init 0, at most 3 per cell, one episode per reference).
Everything runs through lerobot's own rollout (pi0.5 called every 10 steps, per-episode noise as closed_loop_v3.py).
V11 sits in pi0.5's predict_action_chunk: pi0.5 samples its chunk a_pi as usual (branch off), then, as in training
(train_v11_v9.Runner), the action expert runs twice on the clean chunk at flow time read_time -- branch off -> h_off,
branch on with the reference -> h_on -- and the head gives Delta; the chunk becomes a_pi + Delta (normalized space), of
which lerobot executes the first 10 steps. The reference is read as in training: aligned by time (its call nearest to
t), its 50-step chunk from there, the classic scalars, its pi0.5 image tokens and state at that call (computed here by
replaying the reference in its own scene through the same env and preprocessing as the test, cached in <out>/refs).
Modes:
  base     : pi0.5 alone (the branch attached but given nothing: unchanged), two seeds per cell (seed, seed + 4000);
  v11      : the cell's own references (up to 3 per cell);
  v11other : control, a reference of another base task (one per own reference, fixed per cell) -- if this helps as
             much as v11, the branch is not using the reference.
    python closed_loop_v11.py --ckpt runs/v11/train/v9_run2_4090/best.pt --modes v11 base --limit 8 --out runs/v11/closed/smoke
--var_ckpt (10-31, scheme B, train_v11_var.py): the executed correction is Delta shrunk by the variance head with the rule
chosen on validation and stored in that checkpoint; the log keeps both sizes (delta_rms shrunk, delta_raw_rms before).
--refs_dir: reuse the references cached by another run, so the runs share them exactly.
--delta_dims pos (10-31, harm diagnosis): only the position part of Delta is executed; rotation and gripper stay pi0.5's.
--inits, --one_per_cell (10-31, tilt check on more variants): init states to run; one own reference per cell (cycling
through the cell's own by init), one other-task reference, one base seed.
"""
import argparse
import functools
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import train_v11_v9 as V
import v10_oracle_shift2 as O
from v11_check import RowNoise

ROOT = "/root/pace/runs"
CALL, CHUNK, EXEC = V.CALL, V.CHUNK, V.EXEC


def batched(x):
    """A single env's observation (nested dict of arrays) -> the vector env's format (batch axis 0)."""
    if isinstance(x, dict):
        return {k: batched(v) for k, v in x.items()}
    return np.asarray(x)[None]


def make_env(tid, k):
    """The test env exactly as closed_loop_v3.py builds it (lerobot's LIBERO-plus env, default resolution)."""
    c = O.G["env_cfg"]
    fn = O.G["make_fns"](suite=O.G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=O.G["cams"],
                         episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=O.G["gk"],
                         control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    return fn


class V11:
    """The trained branch + head on the frozen pi0.5; wraps policy.predict_action_chunk (see the docstring)."""

    def __init__(self, policy, ckpt, var_ckpt=None, delta_dims="all"):
        from pace.core import CALL_SCALE
        from pace.pi05 import PI05Experience, batch_state, build_adapter, prefix_cache, vis_tokens

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.policy, self.read_time, self.call_scale = policy, float(a["read_time"]), CALL_SCALE
        self.vis_tokens, self.batch_state, self.prefix_cache = vis_tokens, batch_state, prefix_cache
        self.adapter = build_adapter(policy, inject_layers=("every2",), max_refs=a["max_refs"], input_inject=True,
                                     position="post_layer", time_cond=False, encoder="v2",
                                     feature_shapes=((32, 2048), 8), enc_dim=512, enc_layers=2, enc_heads=8,
                                     act_segments=10, retrieval="classic", generic="none", relevance="gate",
                                     ref_isolation=a.get("ref_isolation", "on") == "on")
        self.pace = PI05Experience(policy, self.adapter, runtime_kwargs={"retrieval": "classic"})
        self.pace.attach()
        self.adapter.load_state_dict(ck["adapter"])
        self.head = V.DiffHead().to(self.pace.device)
        self.head.load_state_dict(ck["head"])
        self.adapter.eval()
        self.head.eval()
        self.vh, self.rule, self.W = None, None, None
        self.dims = None if delta_dims == "all" else torch.tensor([1.0, 1, 1, 0, 0, 0, 0], device=self.pace.device)
        if var_ckpt:
            import train_v11_var as W

            self.W = W
            self.vh, self.rule = W.load_var_head(var_ckpt, self.pace.device)
        self.cap = V.Capture(policy.model.action_out_proj)
        z = np.load(f"{ROOT}/v11/api/norm.npz")
        self.mean, self.std = z["mean"], z["std"]
        self.step = ck.get("step")
        self.rows = None  # per batch row: reference dict (vis, state, t, actions) or None (no branch)
        self.calls = 0
        self.stats = None
        orig_pred, orig_reset = policy.predict_action_chunk, policy.reset

        def pred(batch, **kw):
            act = orig_pred(batch, **kw)
            if self.rows is not None and any(r is not None for r in self.rows):
                delta, gate = self.delta(batch, act)
                act = act + delta.to(act.dtype)
                for i, r in enumerate(self.rows):
                    if r is not None:
                        self.stats[i]["delta"].append(float(delta[i, :EXEC].float().pow(2).mean().sqrt()))
                        self.stats[i]["raw"].append(float(self.raw[i, :EXEC].float().pow(2).mean().sqrt()))
                        self.stats[i]["gate"].append(gate)
            self.calls += 1
            return act

        def reset():
            orig_reset()
            self.calls = 0

        policy.predict_action_chunk, policy.reset = pred, reset

    def set_rows(self, rows):
        self.rows = rows
        self.stats = None if rows is None else [dict(delta=[], raw=[], gate=[]) for _ in rows]

    def norm(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def delta(self, obs, a_pi):
        """Delta [B, 50, 7] for the rows with a reference (0 elsewhere), and the relevance gate's batch mean."""
        p, dev, B = self.policy, self.pace.device, a_pi.shape[0]
        t = self.calls * CALL
        lm_cfg = p.model.paligemma_with_expert.paligemma.model.language_model.config
        attn = lm_cfg._attn_implementation  # prefix_cache switches it to eager: restore, pi0.5's own sampling unchanged
        try:
            images, img_masks = p._preprocess_images(obs)
            cur_vis = self.vis_tokens(p.model._embed_images(images, img_masks), (0, 1)).float()
            cur_state = self.batch_state(obs)
            prefix = self.prefix_cache(p, obs)
        finally:
            lm_cfg._attn_implementation = attn
        ref_vis, ref_state, a_ref, rp, valid = [], [], [], [], []
        for r in self.rows:
            if r is None:  # a row without a reference in a mixed batch (not used by main: batches are one mode)
                r, ok = next(x for x in self.rows if x is not None), False
            else:
                ok = True
            j = int(np.argmin(np.abs(r["t"] - t)))  # the reference's call nearest to t (calls only are stored)
            L = len(r["t"])
            ref_vis.append(torch.from_numpy(r["vis"][j].astype(np.float32)))
            ref_state.append(torch.from_numpy(r["state"][j].astype(np.float32)))
            a_ref.append(torch.from_numpy(self.norm(V.chunk(r["actions"], int(r["t"][j])))))
            rp.append([j / max(L - 1, 1), j, L])
            valid.append(ok)
        ref_vis, ref_state = torch.stack(ref_vis).to(dev), torch.stack(ref_state).to(dev)
        rp = torch.tensor(rp, dtype=torch.float32, device=dev)
        sim = (cur_vis * ref_vis).sum(-1).mean(-1)
        call = torch.full((B,), t / CALL, device=dev)
        scalars = torch.stack([rp[:, 0], sim, torch.zeros_like(sim), call / self.call_scale, rp[:, 1] / self.call_scale,
                               rp[:, 2] / self.call_scale], -1)[:, None]
        exp = dict(actions=torch.stack(a_ref).to(dev)[:, None], scalars=scalars,
                   success=torch.ones(B, 1, dtype=torch.long, device=dev),
                   valid=torch.tensor(valid, device=dev)[:, None], ref_vis=ref_vis[:, None], ref_state=ref_state[:, None],
                   cur_vis=cur_vis, cur_state=cur_state)
        x = F.pad(a_pi.float(), (0, p.config.max_action_dim - a_pi.shape[-1]))
        tm = torch.full((B,), self.read_time, device=dev)

        def hidden():
            p.model.denoise_step(prefix_pad_masks=prefix[0], past_key_values=prefix[1], x_t=x, timestep=tm)
            return self.cap.h.float()

        self.pace.set_exp(None)
        h_off = hidden()
        self.adapter.relevance_stats = {}
        self.pace.set_exp(exp)
        try:
            h_on = hidden()
            gate = float(self.adapter.relevance_stats.get("relevance_mean", float("nan")))
        finally:
            self.pace.set_exp(None)
        delta, _ = self.head(h_on, h_off)
        self.raw = delta
        if self.vh is not None:
            delta = self.W.shrink(delta, self.W.var(self.vh, h_on, h_off), self.rule)
        if self.dims is not None:
            delta = delta * self.dims.to(delta.dtype)
        delta = delta * torch.tensor(valid, device=dev, dtype=delta.dtype)[:, None, None]
        return delta, gate


@torch.no_grad()
def prepare_ref(src, out, policy, pre, env_pre, preprocess_observation):
    """Replay a reference in its own scene through the test's env and preprocessing; at each of its calls (t % 10 == 0)
    pi0.5's image tokens (V9 v2-encoder format, as v11_check.py --refvis) and preprocessed state. Cached."""
    from pace.pi05 import batch_state, vis_tokens

    path = f"{out}/refs/t{src['task_id']}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return dict(vis=z["vis"], state=z["state"], t=z["t"], actions=src["actions"])
    fn = make_env(src["task_id"], src["init_index"])
    env = fn.func(int(src["init_index"]), **fn.keywords)
    vis, st, ts = [], [], []
    try:
        obs, _ = env.reset(seed=int(src["seed"]))
        for t, a in enumerate(src["actions"]):
            if t % CALL == 0:
                o = preprocess_observation(batched(obs))
                o["task"] = [src["language"]]
                o = pre(env_pre(o))
                images, img_masks = policy._preprocess_images(o)
                vis.append(vis_tokens(policy.model._embed_images(images, img_masks), (0, 1)).half().cpu().numpy()[0])
                st.append(batch_state(o).float().cpu().numpy()[0])
                ts.append(t)
            obs, _, term, trunc, info = env.step(a)
            if info.get("is_success"):
                break
    finally:
        env.close()
    os.makedirs(f"{out}/refs", exist_ok=True)
    np.savez(path, vis=np.stack(vis), state=np.stack(st), t=np.array(ts))
    return dict(vis=np.stack(vis), state=np.stack(st), t=np.array(ts), actions=src["actions"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/v11/train/v9_run2_4090/best.pt")
    ap.add_argument("--out", default=f"{ROOT}/v11/closed/run1")
    ap.add_argument("--modes", nargs="+", default=["v11", "v11other", "base"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: the first episodes only")
    ap.add_argument("--var_ckpt", default=None, help="variance head (train_v11_var.py): shrink the correction")
    ap.add_argument("--refs_dir", default=None, help="where the references are cached (default --out)")
    ap.add_argument("--delta_dims", choices=["all", "pos"], default="all", help="pos: execute only Delta's position part")
    ap.add_argument("--targets", default=None, help="JSON of variants (select_noncam.py) instead of the 20 camera ones "
                    "(10-30: the non-camera test, runs/rectest/v10_shift/noncam_targets.json)")
    ap.add_argument("--inits", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--one_per_cell", action="store_true", help="one own reference, one other-task, one base seed per cell")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from lerobot.envs.libero import _LazyAsyncVectorEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.scripts.lerobot_eval import rollout
    from pace.eval_suite import FixedTasks
    from v11_check import load_pi05

    O.worker_init()
    policy, noise, pre, env_pre, _, _ = load_pi05()  # pi0.5, per-row noise, processors (as V11's training data)
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors

    _, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                       preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                               "rename_observations_processor": {"rename_map": {}}})
    _, env_post = make_env_pre_post_processors(O.G["env_cfg"], policy.config)
    v11 = V11(policy, args.ckpt, args.var_ckpt, args.delta_dims)
    print(f"[v11] {args.ckpt} (step {v11.step}), read_time {v11.read_time}, var_ckpt {args.var_ckpt}, delta_dims {args.delta_dims}, "
          f"rule {None if v11.rule is None else v11.rule['signal']}", flush=True)

    targets = [json.loads(l) for l in open(f"{ROOT}/rectest/v10_shift/base_targets/episodes.jsonl")]
    cams = sorted({r["task_id"]: dict(r, basename=r["name"].split("_view_")[0]) for r in targets}.items())
    if args.targets:  # other categories (as closed_loop_v3.py): base task = the variant's stem, pi0.5 gets the variant's
        # own instruction (language variants are reworded)
        cams = sorted((x["task_id"], dict(basename=x["base"], base=x["language"], category=x["category"]))
                      for x in json.load(open(args.targets)))
    sources = {}
    for r in map(json.loads, open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")):
        if r["success"]:
            sources[r["task_id"]] = dict(task_id=int(r["task_id"]), init_index=int(r["init"]), seed=int(r["seed"]),
                                         base=r["base"], language=r["language"],
                                         actions=np.load(f"{ROOT}/v10b/noise_sources/{r['actions']}").astype(np.float64))
    entries = []
    for tid, r in cams:
        own = [s for s in sources.values() if s["base"] == r["basename"]][:3]
        others = [s for s in sources.values() if s["base"] != r["basename"]]
        for k in args.inits:
            seed = 1000 + 7 * tid + k
            rng = np.random.default_rng([tid, k])
            for i, s in enumerate([own[(k - 1) % len(own)]] if args.one_per_cell and own else own):
                if "v11" in args.modes:
                    entries.append(dict(mode="v11", task_id=tid, init=k, seed=seed, language=r["base"], src=s, category=r.get("category"),
                                        key=f"v11_t{tid}_i{k}_s{s['task_id']}"))
                if "v11other" in args.modes:
                    o = others[int(rng.integers(len(others)))]
                    entries.append(dict(mode="v11other", task_id=tid, init=k, seed=seed, language=r["base"], src=o, category=r.get("category"),
                                        key=f"v11other_t{tid}_i{k}_n{i}_s{o['task_id']}"))
            if "base" in args.modes:
                for sd in (seed,) if args.one_per_cell else (seed, seed + 4000):
                    entries.append(dict(mode="base", task_id=tid, init=k, seed=sd, language=r["base"], src=None, category=r.get("category"),
                                        key=f"base_t{tid}_i{k}_seed{sd}"))
    log_path = f"{args.out}/episodes.jsonl"
    done = {json.loads(l)["key"] for l in open(log_path)} if os.path.exists(log_path) else set()
    todo = [e for e in entries if e["key"] not in done]
    if args.limit:  # smoke: the first episodes of each mode
        todo = [e for m in args.modes for e in [x for x in todo if x["mode"] == m][:args.limit]]
    print(f"[closed] {len(entries)} episodes ({len(todo)} to do): "
          + ", ".join(f"{m} {sum(e['mode'] == m for e in entries)}" for m in args.modes), flush=True)
    refs = {}
    for e in todo:
        s = e["src"]
        if s is not None and s["task_id"] not in refs:
            refs[s["task_id"]] = prepare_ref(s, args.refs_dir or args.out, policy, pre, env_pre, preprocess_observation)
    print(f"[closed] {len(refs)} references ready", flush=True)

    log = open(log_path, "a")
    t0 = time.time()
    for m in args.modes:  # one mode per batch: a base batch never touches the branch
        tm = [e for e in todo if e["mode"] == m]
        for b0 in range(0, len(tm), args.batch):
            batch = tm[b0:b0 + args.batch]
            fns = []
            for e in batch:
                fn = make_env(e["task_id"], e["init"])
                fns.append(functools.partial(fn.func, e["init"], **fn.keywords))
            v11.set_rows(None if m == "base" else [refs[e["src"]["task_id"]] for e in batch])
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
                n_calls = (T + CALL - 1) // CALL  # calls made while this episode was running
                st = v11.stats[i] if v11.stats is not None and m != "base" else None
                log.write(json.dumps(dict(
                    key=e["key"], mode=m, task_id=e["task_id"], init=e["init"], seed=e["seed"], category=e["category"],
                    source=e["src"]["task_id"] if e["src"] else None, success=bool(succ[i]), steps=T,
                    delta_rms=float(np.mean(st["delta"][:n_calls])) if st and st["delta"] else None,
                    delta_raw_rms=float(np.mean(st["raw"][:n_calls])) if st and st["raw"] else None,
                    gate=float(np.nanmean(st["gate"][:n_calls])) if st and st["gate"] else None,
                    ckpt=args.ckpt, ckpt_step=v11.step, var_ckpt=args.var_ckpt, delta_dims=args.delta_dims, gpu=torch.cuda.get_device_name(0))) + "\n")
            log.flush()
            print(f"[closed] {m} {b0 + len(batch)}/{len(tm)} ({time.time() - t0:.0f}s), batch success "
                  f"{sum(succ)}/{len(batch)}", flush=True)
    v11.set_rows(None)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
