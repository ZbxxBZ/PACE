"""
V11b closed loop (11-02): closed_loop_v11.py with V11b's model in pi0.5's predict_action_chunk; same targets, seeds, references, modes and
arguments. The chunk becomes a_pi + g Delta (train_v11b.Runner.step). The reference step follows the checkpoint's --align: pma runs the
state-only alignment online, one step per pi0.5 call, on the env's raw hand position and gripper (taken before pi0.5's preprocessing), so
references are cached with theirs in <refs_dir>/refs_b; time is V11's nearest call. The log's gate is the mean of g.
    python closed_loop_v11b.py --ckpt runs/v11b/train/pma_full/best.pt --modes v11 v11other base --out runs/v11b/closed/run1
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

import closed_loop_v11 as C
import pace.eval_suite as ES
import train_v11b as B

V, ROOT = C.V, C.ROOT
CALL, EXEC = C.CALL, C.EXEC
SEEN = {}


class Aligner:
    """train_v11b.pma, one call at a time."""

    def __init__(self, r):
        self.rp, self.rg, self.J = r["pos"], r["grip"], None

    def step(self, p, g):
        c = 1.0 - np.exp(-((self.rp - p) ** 2).sum(-1) / B.SP ** 2 - ((self.rg - g) ** 2).sum(-1) / B.SG ** 2)
        if self.J is None:
            self.J = c
        else:
            best = np.full(len(c), np.inf)
            for d in range(min(B.VMAX + 1, len(c))):
                s = self.J if d == 0 else np.r_[np.full(d, np.inf), self.J[:-d]]
                best = np.minimum(best, s + B.GAMMA * abs(d - 1))
            self.J = c + best
        return int(self.J.argmin())


class Tap(ES.FixedTasks):
    """FixedTasks that also keeps the raw hand position and gripper of the current call."""

    def __call__(self, observation):
        st = observation["observation.robot_state"]
        SEEN["pos"], SEEN["grip"] = st["eef"]["pos"].cpu().numpy(), st["gripper"]["qpos"].cpu().numpy()
        return super().__call__(observation)


class V11b:
    """Same interface as closed_loop_v11.V11."""

    def __init__(self, policy, ckpt, var_ckpt=None, delta_dims="all"):
        from pace.core import CALL_SCALE
        from pace.pi05 import PI05Experience, batch_state, build_adapter, prefix_cache, vis_tokens

        assert var_ckpt is None and delta_dims == "all"
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        self.policy, self.read_time, self.call_scale, self.align = policy, float(a["read_time"]), CALL_SCALE, a["align"]
        self.vis_tokens, self.batch_state, self.prefix_cache = vis_tokens, batch_state, prefix_cache
        self.adapter = build_adapter(policy, inject_layers=("every2",), max_refs=a["max_refs"], input_inject=True,
                                     position="post_layer", time_cond=False, encoder="v2",
                                     feature_shapes=((32, 2048), 8), enc_dim=512, enc_layers=2, enc_heads=8,
                                     act_segments=10, retrieval="classic", generic="none", relevance="gate",
                                     ref_isolation=a.get("ref_isolation", "off") == "on")
        self.pace = PI05Experience(policy, self.adapter, runtime_kwargs={"retrieval": "classic"})
        self.pace.attach()
        self.adapter.load_state_dict(ck["adapter"])
        dev = self.pace.device
        self.head, self.cmp, self.gate = B.DeltaHead().to(dev), B.Compare().to(dev), B.Gate().to(dev)
        self.head.load_state_dict(ck["head"]); self.cmp.load_state_dict(ck["cmp"]); self.gate.load_state_dict(ck["gate"])
        for m in (self.adapter, self.head, self.cmp, self.gate):
            m.eval()
        self.cap = C.V.Capture(policy.model.action_out_proj)
        z = np.load(f"{ROOT}/v11/api/norm.npz")
        self.mean, self.std = z["mean"], z["std"]
        self.step, self.rule = ck.get("step"), None
        self.rows, self.al, self.stats, self.calls = None, None, None, 0
        orig_pred, orig_reset = policy.predict_action_chunk, policy.reset

        def pred(batch, **kw):
            act = orig_pred(batch, **kw)
            if self.rows is not None:
                delta, g, raw = self.delta(batch, act)
                act = act + delta.to(act.dtype)
                for i in range(len(self.rows)):
                    self.stats[i]["delta"].append(float(delta[i, :EXEC].float().pow(2).mean().sqrt()))
                    self.stats[i]["raw"].append(float(raw[i, :EXEC].float().pow(2).mean().sqrt()))
                    self.stats[i]["gate"].append(float(g[i]))
            self.calls += 1
            return act

        def reset():
            orig_reset()
            self.calls = 0
            self.al = None if self.rows is None else [Aligner(r) for r in self.rows]

        policy.predict_action_chunk, policy.reset = pred, reset

    def set_rows(self, rows):
        self.rows = rows
        self.stats = None if rows is None else [dict(delta=[], raw=[], gate=[]) for _ in rows]
        self.al = None if rows is None else [Aligner(r) for r in rows]

    def norm(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def delta(self, obs, a_pi):
        p, dev, B_ = self.policy, self.pace.device, a_pi.shape[0]
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
        ref_vis, ref_state, a_ref, rp = [], [], [], []
        for i, r in enumerate(self.rows):
            j = self.al[i].step(SEEN["pos"][i], SEEN["grip"][i]) if self.align == "pma" else int(np.argmin(np.abs(r["t"] - t)))
            L = len(r["t"])
            ref_vis.append(torch.from_numpy(r["vis"][j].astype(np.float32)))
            ref_state.append(torch.from_numpy(r["state"][j].astype(np.float32)))
            a_ref.append(torch.from_numpy(self.norm(C.V.chunk(r["actions"], int(r["t"][j])))))
            rp.append([j / max(L - 1, 1), j, L])
        ref_vis, ref_state = torch.stack(ref_vis).to(dev), torch.stack(ref_state).to(dev)
        rp = torch.tensor(rp, dtype=torch.float32, device=dev)
        sim = (cur_vis * ref_vis).sum(-1).mean(-1)
        call = torch.full((B_,), t / CALL, device=dev)
        scalars = torch.stack([rp[:, 0], sim, torch.zeros_like(sim), call / self.call_scale, rp[:, 1] / self.call_scale,
                               rp[:, 2] / self.call_scale], -1)[:, None]
        exp = dict(actions=torch.stack(a_ref).to(dev)[:, None], scalars=scalars, success=torch.ones(B_, 1, dtype=torch.long, device=dev),
                   valid=torch.ones(B_, 1, dtype=torch.bool, device=dev), ref_vis=ref_vis[:, None], ref_state=ref_state[:, None],
                   cur_vis=cur_vis, cur_state=cur_state)
        x = F.pad(a_pi.float(), (0, p.config.max_action_dim - a_pi.shape[-1]))
        tm = torch.full((B_,), self.read_time, device=dev)

        def hidden():
            p.model.denoise_step(prefix_pad_masks=prefix[0], past_key_values=prefix[1], x_t=x, timestep=tm)
            return self.cap.h.float()

        self.pace.set_exp(None)
        h_off = hidden()
        self.pace.set_exp(exp)
        try:
            h_on = hidden()
        finally:
            self.pace.set_exp(None)
        d_b, _ = self.head(h_on, h_off)
        c = self.cmp.tokens(cur_vis, ref_vis)
        raw = d_b + self.cmp(c, h_off)
        g = self.gate(c, cur_state.float() - ref_state.float())
        return g[:, None, None] * raw, g, raw


@torch.no_grad()
def prepare_ref(src, out, policy, pre, env_pre, preprocess_observation):
    """closed_loop_v11.prepare_ref, keeping the reference's raw hand position and gripper at its calls."""
    from pace.pi05 import batch_state, vis_tokens

    path = f"{out}/refs_b/t{src['task_id']}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return dict(vis=z["vis"], state=z["state"], t=z["t"], pos=z["pos"], grip=z["grip"], actions=src["actions"])
    fn = C.make_env(src["task_id"], src["init_index"])
    env = fn.func(int(src["init_index"]), **fn.keywords)
    vis, st, ts, pos, grip = [], [], [], [], []
    try:
        obs, _ = env.reset(seed=int(src["seed"]))
        for t, a in enumerate(src["actions"]):
            if t % CALL == 0:
                o = preprocess_observation(C.batched(obs))
                o["task"] = [src["language"]]
                o = pre(env_pre(o))
                images, img_masks = policy._preprocess_images(o)
                vis.append(vis_tokens(policy.model._embed_images(images, img_masks), (0, 1)).half().cpu().numpy()[0])
                st.append(batch_state(o).float().cpu().numpy()[0])
                ts.append(t)
                pos.append(np.array(obs["robot_state"]["eef"]["pos"]))
                grip.append(np.array(obs["robot_state"]["gripper"]["qpos"]))
            obs, _, term, trunc, info = env.step(a)
            if info.get("is_success"):
                break
    finally:
        env.close()
    os.makedirs(f"{out}/refs_b", exist_ok=True)
    r = dict(vis=np.stack(vis), state=np.stack(st), t=np.array(ts), pos=np.stack(pos), grip=np.stack(grip))
    np.savez(path, **r)
    return dict(r, actions=src["actions"])


if __name__ == "__main__":
    C.V11, C.prepare_ref, ES.FixedTasks = V11b, prepare_ref, Tap
    C.main()
