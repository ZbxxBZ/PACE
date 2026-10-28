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
--targets runs/rectest/v10_shift/noncam_targets.json (10-24): the non-camera test (select_noncam.py).
Mode v10t (10-24): the V10-2 trial -- pi0.5 first, experience takes over on an observable failure (V10Env._takeover).
    python closed_loop_v2.py --modes v10t --out runs/v10b/closed/run2    # base episodes of run2 are reused
Mode v10p (10-25, closed_loop_v2p.py; registered before running, replaces v10t): per-phase takeover. The d carryover
check of run2 (d_carryover.py) found the learned d drifting toward the previous phase's shift after the first phase
(failed episodes, second grasp: 62% of calls closer to the previous phase's d), so a follow now covers ONE reference
phase and hands back to pi0.5 SETTLE steps after that phase's grasp / release (the last phase: after the reference
ends + HOLD). Step 0: follow the first phase if P(follow) - P(pi0.5) > margin_start (0.3). Afterwards pi0.5 acts; on an
empty grasp / being stuck the heads are asked at the start of the reference phase pi0.5 has reached and that phase is
followed if the difference > margin (0); at most MAX_TAKE takeovers per episode; d is re-estimated from scratch at
each takeover and every call within it.
    python closed_loop_v2p.py --modes v10p --out runs/v10b/closed/run2
Mode v10q (10-25, closed_loop_v2q.py; V10-2b, no training): V10-1's start rule (follow the first phase if
P(follow) - P(pi0.5) > margin, 0) and, at the end of every followed phase, the same two heads asked at the start of the
next reference phase: difference > margin -> follow that phase too, else hand back to pi0.5 (the v10p rule triggers
then stay available, at most MAX_TAKE takeovers). Differs from V10-1 only by the option to hand back at phase ends.
    python closed_loop_v2q.py --modes v10q --batch 6 --out runs/v10b/closed/run2
closed_loop_explore.py (10-25): data for the V10-3 value head, on the TRAINING scenes only (the 10 original libero_10
tasks, init states 6..49 = training 6..44 / validation 45..49; never the test variants). Scene augmentation 0 of
augment.scene_aug for both pi0.5's observations (AugEnv) and the network's 224 render; one reference per scene: a
training reference of the same task and split (runs/v10b/refs, features from runs/v10b/feats/refs). Mode explore:
the same per-phase machinery as v10q, but every decision is random (seeded per episode) so that both options get
outcome labels -- step 0: follow the first phase with prob. 0.5; end of a followed phase: follow the next one with
prob. 0.5; pi0.5 in control: after each of its grasps / releases take over the next phase with prob. 0.3, on an
empty grasp / being stuck with prob. 0.5 (at most MAX_TAKE takeovers). At every decision point the network's inputs
(current features, state, reference rows) and outputs are saved (<out>/decisions/<key>.npz); the label is the
episode's outcome (episodes.jsonl).
    python closed_loop_explore.py --modes explore --train_scenes --batch 6 --out runs/v10b/closed/explore
closed_loop_v3.py, mode v10v (10-25): V10-3 -- the same decision points as explore (step 0, end of a followed phase,
after each of pi0.5's grasps / releases, empty grasp / stuck), each decided by the value head (train_value.py):
follow / take over iff P(take) - P(cont) > --margin_v; d from the V10-3 backbone (--ckpt run3, hand offset relative to
the start of the followed segment; 0 at a decision point).
    python closed_loop_v3.py --modes v10v --ckpt runs/v10b/train/run3/best.pt --value runs/v10b/value/run1/value.pt \
        --batch 6 --out runs/v10b/closed/run2
closed_loop_v32.py (10-27): V10-3-2, soft follow (user naming: V10-3-1 = mode v10v, hard follow; V10-3-2 = soft). Same
decisions as v10v (value head at step 0 -- now taken in reset(), before pi0.5's first call -- and at phase ends), but
while following the wrapper does not replace pi0.5's action: at every pi0.5 call (every CALL steps) it writes the
registered target chunk (CHUNK reference steps from the current reference time, a_ref + the current hand error spread
over the first SOFT_SPREAD steps, env units) to /dev/shm, and in the main process pi0.5's flow matching starts that row
at time t0 from x = t0 * noise + (1 - t0) * normalize(target) instead of from pure noise (SoftCtl / soft_euler; only
round(10 * t0) of the 10 Euler steps run). Mode v10s<NN>: t0 = NN / 100; v10s0 executes the target chunk itself
(chunked hard follow, the control for the chunking); rows that are not following are sampled exactly as before.
    python closed_loop_v32.py --modes v10s20 v10s40 --ckpt runs/v10b/train/run1/best.pt \
        --value runs/v10b/value/low_run1.npz --batch 6 --out runs/v10b/closed/soft
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
EMPTY = 0.0025  # finger opening (q0 - q1) 15 steps after a close command: below = empty grasp (closed on nothing 0.0012,
#                 holding the thinnest object >= 0.0035; measured 10-24)
GRASP_CHECK, STUCK_X, STUCK_ADD, RECHECK = 15, 2.0, 20, 50
SETTLE, MAX_TAKE = 15, 3  # v10p: steps after the phase's gripper event before handing back; takeovers per episode
_CACHE = {}
CHUNK, SOFT_SPREAD = 50, 5  # pi0.5's chunk length; steps over which a target chunk spreads the current hand error
SOFT_DIR = f"/dev/shm/pace_soft_{os.getpid()}"  # the main process's pid; workers get their file path in cfg


def soft_t0(mode):
    """V10-3-2 modes v10s<NN>: soft follow starting pi0.5's flow at t0 = NN / 100; None for the other modes."""
    return int(mode[4:]) / 100.0 if mode.startswith("v10s") else None


def is_v3(mode):
    """V10-3's decision logic: V10-3-1 (v10v, hard follow) and V10-3-2 (v10s<NN>, soft follow)."""
    return mode == "v10v" or mode.startswith("v10s")


class SoftCtl:
    """Main-process side of V10-3-2: per batch row, the target chunk its wrapper wrote for the current pi0.5 call
    (checked by episode key and step), normalized as pi0.5's actions (MEAN_STD; the env postprocessor is empty, so
    env action = model action * std + mean), padded with zeros to max_action_dim."""

    def __init__(self, mean, std):
        self.mean, self.std = mean, std
        self.keys, self.paths, self.calls = [], [], 0
        self.stats = {"guided": 0, "stale": 0}

    def new_batch(self, keys, paths):
        self.keys, self.paths, self.calls = list(keys), list(paths), 0
        self.stats = {"guided": 0, "stale": 0}  # per batch
        for p in paths:
            if os.path.exists(p):
                os.remove(p)

    def fetch(self, noise):
        b = noise.shape[0]
        tgt, t0 = torch.zeros_like(noise), torch.ones(b, dtype=torch.float32, device=noise.device)
        want = CALL * self.calls
        for i, p in enumerate(self.paths[:b]):
            if not os.path.exists(p):
                continue
            with np.load(p) as z:
                key, t, t0_i, a = str(z["key"]), int(z["t"]), float(z["t0"]), np.array(z["target"])
            if key != self.keys[i] or t != want:
                self.stats["stale"] += 1
                continue
            a = torch.from_numpy(a).float().to(noise.device)
            tgt[i, :a.shape[0], :a.shape[1]] = (a - self.mean.to(noise.device)) / self.std.to(noise.device)
            t0[i] = t0_i
            self.stats["guided"] += 1
        return tgt, t0


def soft_euler(denoise_fn, noise, num_steps=None, **kw):
    """lerobot's euler_integrate (NOISE_AT_ONE, uniform grid) with per-row start times: a guided row starts at
    t0 from t0 * noise + (1 - t0) * target and is integrated only on the grid points tau <= t0; unguided rows
    (t0 = 1) get exactly the original updates."""
    ctl = _CACHE.get("soft")
    if (ctl is None or kw.get("rtc_enabled") or kw.get("hard_prefix") is not None
            or kw.get("time_grid") is not None or kw.get("velocity_scale") is not None):
        return _CACHE["euler"](denoise_fn, noise, num_steps, **kw)
    tgt, t0 = ctl.fetch(noise)
    ctl.calls += 1
    if bool((t0 >= 1).all()):
        return _CACHE["euler"](denoise_fn, noise, num_steps, **kw)
    t0b = t0[:, None, None]
    x = torch.where(t0b < 1, t0b * noise + (1 - t0b) * tgt, noise)
    dt = -1.0 / num_steps
    for step in range(num_steps):
        tau = 1.0 + step * dt
        v = denoise_fn(x, torch.full((noise.shape[0],), tau, dtype=torch.float32, device=noise.device))
        x = torch.where(t0b >= tau - 1e-6, x + dt * v, x)
    return x


def install_soft(post):
    """Patch pi0.5's integrator (module global used by PI05Pytorch.sample_actions) and read the action stats."""
    import lerobot.policies.pi05.modeling_pi05 as MP

    st = next(s_.stats for s_ in post.steps if getattr(s_, "stats", None))["action"]
    mean = torch.as_tensor(np.asarray(st["mean"], dtype=np.float32))
    std = torch.as_tensor(np.asarray(st["std"], dtype=np.float32))
    _CACHE["euler"] = MP.euler_integrate
    _CACHE["soft"] = SoftCtl(mean, std)
    MP.euler_integrate = soft_euler
    os.makedirs(SOFT_DIR, exist_ok=True)
    return _CACHE["soft"]


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
        _CACHE["state"] = a.get("state", "abs")
        if cfg.get("value") and cfg["value"].endswith(".npz"):  # low-dim logistic value head (fit_value_low.py)
            _CACHE["vlow"] = dict(np.load(cfg["value"]))
        elif cfg.get("value"):
            import train_value as TV

            vk = torch.load(cfg["value"], weights_only=False)
            head = TV.ValueHead(vk["dim_h"]).cuda().eval()
            head.load_state_dict(vk["head"])
            _CACHE["v"] = (head, torch.tensor(vk["temps"]).cuda())
    return _CACHE["m"]


def corrupt(img, level):
    """LIBERO-plus's sensor-noise corruption of the agentview image (env_wrapper.py: applied to the observation only),
    so the network's 224 render sees what pi0.5 sees on a Sensor Noise variant (10-24, non-camera test)."""
    if not level:
        return img
    from PIL import Image
    from libero.libero.envs import env_wrapper as W

    pil = Image.fromarray(img)
    for hi, fn, off in ((10, W.motion_blur, 0), (20, W.gaussian_blur, 10), (30, W.zoom_blur, 20), (40, W.fog, 30),
                        (50, W.glass_blur, 40)):
        if level <= hi:
            return np.ascontiguousarray(np.asarray(fn(pil, severity=level - off)).astype(np.uint8))
    return img


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
        # v10t: reference time = tau0 + (t - t_start) while following; pi0.5's grasp / release progress otherwise
        self.tau0 = self.t_start = 0
        self.k, self.holding, self.cmd, self.pend, self.close_t, self.last_ev = 0, False, False, 0, None, 0
        self.taken, self.next_check, self.triggers = False, 0, []
        self.from_start, self.finished = False, False  # followed from step 0; a follow has ended (no more takeovers)
        self.seg_ph, self.n_take, self.segs = 0, 0, []  # v10p: phase being followed, takeovers so far, follow segments
        for k_ in ("erng", "dec", "_event"):  # explore: per-episode state
            self.__dict__.pop(k_, None)
        self._started, self.soft = False, soft_t0(self.cfg["mode"])
        self._soft_tgt, self._soft_t, self._soft_dev = None, 0, []
        if self.soft is not None and not self.written:  # V10-3-2: decide before pi0.5's first call, guide it
            self._takeover_phase(None, cont=True, explore=True)
            self._soft_write()
        return obs, info

    @torch.no_grad()
    def _infer(self, inner, tau=None, at_start=False):
        c, src = self.cfg, self.cfg["src"]
        model, tower, mean, comp, d_scale = _models(c)
        im = render224(inner)
        im["agentview"] = corrupt(im["agentview"], getattr(self.env.unwrapped._env, "noise", 0))
        if c.get("aug") is not None:
            import augment as A

            if not hasattr(self, "prng"):
                self.prng = np.random.default_rng(c["aug"]["seed"] + 1)
            im = {k_: A.pixels(v, c["aug"], self.prng) for k_, v in im.items()}
        cur = encode(tower, mean, comp, [im["agentview"], im["robot0_eye_in_hand"]])[None]  # [1, 2, 256, D]
        t = self.t if tau is None else tau
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
        off = eef - src["eef"][j]
        if _CACHE.get("state") == "phase":  # V10-3 backbone: relative to the offset at the segment start, 0 at a decision
            off = np.zeros(3) if at_start or not hasattr(self, "seg_off0") else off - self.seg_off0
        state = np.r_[off, grip, src["grip"][j], min((t - s) / max(e - s, 1), 1.5), t / 600.0]
        st = torch.tensor(state, dtype=torch.float32).cuda()[None]
        lang = torch.tensor(np.stack(c["lang"]), dtype=torch.float32).cuda()[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, lf, lb, h = model(feats, st, lang, torch.zeros(1, dtype=torch.bool).cuda(), d_scale, return_h=True)
        self._h = h.float()
        self._x = torch.stack([torch.sigmoid(lf.float()), torch.sigmoid(lb.float()), 100 * d.float().norm(dim=1)], 1)
        self._last = dict(cur=cur[0].half().cpu().numpy(), state=state.astype(np.float32), rows=np.array(rows), tau=int(t),
                          eef=eef.astype(np.float32), grip=grip.astype(np.float32))
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
        elif c["mode"] == "v10t":
            a = self._takeover(a)
        elif c["mode"] == "v10p":
            a = self._takeover_phase(a)
        elif c["mode"] == "v10q":
            a = self._takeover_phase(a, cont=True)
        elif c["mode"] == "explore" or is_v3(c["mode"]):
            a = self._takeover_phase(a, cont=True, explore=True)
        obs, r, term, trunc, info = self.env.step(a)
        self.t += 1
        if (self.soft is not None and self.t % CALL == 0 and not (term or trunc or info.get("is_success"))
                and not self.written):
            self._soft_write()
        if self.soft is not None and (term or trunc or info.get("is_success")) and os.path.exists(self.cfg["soft_path"]):
            os.remove(self.cfg["soft_path"])  # the episode is over: no target left behind for later calls
        if term or trunc or info.get("is_success"):
            self._dump(bool(info.get("is_success")))
            self.written = True
        return obs, r, term, trunc, info

    def _dump(self, success):
        if (self.cfg["mode"] in ("v10", "v10t", "v10p", "v10q", "explore") or is_v3(self.cfg["mode"])) \
                and self.cfg.get("log") and not self.written:
            dev = np.array(getattr(self, "_soft_dev", []) or [[0.0, 0.0]])
            with open(self.cfg["log"], "w") as f:
                json.dump(dict(soft=dict(t0=self.soft, n=len(getattr(self, "_soft_dev", [])),
                                         dev_pos=round(float(dev[:, 0].mean()), 4),
                                         grip_flip=round(float(dev[:, 1].mean()), 4))
                               if getattr(self, "soft", None) is not None else None,
                               follow=bool(self.follow) or self.taken or self.from_start,
                               from_start=bool(self.follow) if self.cfg["mode"] == "v10" else self.from_start,
                               taken=self.taken,
                               triggers=self.triggers, segments=self.segs, calls=self.log, steps=self.t,
                               success=success), f)

    def _follow_action(self, inner, tau):
        src = self.cfg["src"]
        n = len(src["actions"])
        tt = min(tau, n - 1)
        ar = src["actions"][tt].copy() if tau < n else np.r_[np.zeros(6), src["actions"][-1][6]]
        hand = np.array(inner.sim.data.site_xpos[inner.robots[0].eef_site_id])
        ar[:3] = np.clip(ar[:3] + (src["path"][tt] + np.r_[self.d, 0.0] - hand) / 0.05, -1.0, 1.0)
        return ar

    def _takeover(self, a):
        """V10-2 trial (registered 10-24 before running): at step 0 follow from the start only if P(follow) - P(pi0.5)
        > margin_start (0.3, validation: damage 2.6%); otherwise pi0.5 acts, and on an observable failure -- an empty
        grasp (opening < EMPTY 15 steps after a close command), being stuck (no grasp / release for STUCK_X x the
        reference phase's length + STUCK_ADD steps; re-checked every RECHECK steps) -- the heads are asked at the start
        of the reference phase pi0.5 has reached (= number of grasps + releases done), and the reference is followed
        from there if P(follow) - P(pi0.5) > margin (0). One takeover at most; after the reference ends (+ HOLD) pi0.5
        acts again. Uses V10-1b's network unchanged (heads not trained for mid-episode states)."""
        c, src = self.cfg, self.cfg["src"]
        inner = self.env.unwrapped._env.env
        n, phases = len(src["actions"]), src["phases"]
        if self.t == 0:
            d, pf, pb = self._infer(inner, 0)
            self.follow, self.d = (pf - pb) > c["margin_start"], d
            self.from_start = bool(self.follow)
            self.log.append(dict(t=0, tau=0, pf=round(pf, 3), pb=round(pb, 3), d=[round(float(x), 4) for x in d]))
            self._dump(None)
        if self.follow:
            tau = self.tau0 + (self.t - self.t_start)
            if tau >= n + HOLD:
                self.follow, self.finished = False, True  # reference done: pi0.5 again, no further takeover
                return a
            if (self.t - self.t_start) % CALL == 0 and tau < n and self.t > 0:
                self.d = self._infer(inner, tau)[0]
            return self._follow_action(inner, tau)
        if self.taken or self.finished or self.follow is None:
            return a
        # pi0.5 in control: track its grasp / release progress from its own gripper commands
        closed = a[6] > 0
        self.pend = self.pend + 1 if closed != self.cmd else 0
        if self.pend >= 3:
            self.cmd, self.pend = closed, 0
            if closed:
                self.close_t = self.t
            elif self.holding:
                self.holding, self.k, self.last_ev = False, self.k + 1, self.t  # a release
                if explore:
                    self._event = True
        trig = None
        if self.cmd and self.close_t is not None and self.t - self.close_t == GRASP_CHECK:
            q = inner.sim.data.qpos[inner.robots[0]._ref_gripper_joint_pos_indexes]
            if float(q[0] - q[1]) >= EMPTY:
                self.holding, self.k, self.last_ev = True, self.k + 1, self.t  # a grasp
                if explore:
                    self._event = True
            else:
                trig = "empty"
        if explore and getattr(self, "_event", False) and trig is None:
            self._event = False
            if self.k < len(phases):
                trig = "event"
        ph = min(self.k, len(phases) - 1)
        length = max(phases[ph][1] - phases[ph][0], CALL)
        if trig is None and self.t >= self.next_check and self.t - self.last_ev > STUCK_X * length + STUCK_ADD:
            trig = "stuck"
        if trig is None:
            return a
        self.next_check = self.t + RECHECK
        tau0 = phases[ph][0]
        d, pf, pb = self._infer(inner, tau0, at_start=True)
        go = (pf - pb) > c["margin"]
        if explore:
            go = self._decide(trig, ph, go, 0.3 if trig == "event" else 0.5, pf, pb, d)
        self.triggers.append(dict(t=self.t, why=trig, k=self.k, tau0=int(tau0), pf=round(pf, 3), pb=round(pb, 3), go=bool(go)))
        self._dump(None)
        if not go:
            return a
        self.follow, self.taken, self.tau0, self.t_start, self.d = True, True, int(tau0), self.t, d
        return self._follow_action(inner, int(tau0))

    def _soft_write(self):
        """V10-3-2: at a pi0.5 call while following, the target chunk for this call (else remove the row's file)."""
        c = self.cfg
        path = c["soft_path"]
        if not self.follow:
            if os.path.exists(path):
                os.remove(path)
            self._soft_tgt = None
            return
        src = c["src"]
        inner = self.env.unwrapped._env.env
        n = len(src["actions"])
        tau = self.tau0 + (self.t - self.t_start)
        if self.t > self.t_start and tau < n:  # d re-estimated at every call (hard follow: every CALL steps too)
            self.d = self._infer(inner, tau)[0]
        hand = np.array(inner.sim.data.site_xpos[inner.robots[0].eef_site_id])
        err = (src["path"][min(tau, n - 1)] + np.r_[self.d, 0.0] - hand) / 0.05
        tgt = np.zeros((CHUNK, 7))
        for k in range(CHUNK):
            tk = tau + k
            ar = src["actions"][min(tk, n - 1)].copy() if tk < n else np.r_[np.zeros(6), src["actions"][-1][6]]
            if k < SOFT_SPREAD:
                ar[:3] += err / SOFT_SPREAD
            ar[:3] = np.clip(ar[:3], -1.0, 1.0)
            tgt[k] = ar
        tmp = path[:-4] + "_tmp.npz"
        np.savez(tmp, key=c["key"], t=self.t, t0=self.soft, target=tgt)
        os.replace(tmp, path)
        self._soft_tgt, self._soft_t = tgt, self.t
        self.log.append(dict(t=self.t, tau=int(tau), d=[round(float(x), 4) for x in self.d], soft=1))

    def _soft_action(self, a):
        """V10-3-2: execute pi0.5's action; record how far it is from the target (position, gripper sign)."""
        k = self.t - self._soft_t
        if self._soft_tgt is not None and 0 <= k < CHUNK:
            g = self._soft_tgt[k]
            self._soft_dev.append([float(np.abs(a[:3] - g[:3]).mean()), float((a[6] > 0) != (g[6] > 0))])
        return a

    def _seg_end(self, ph):
        """Reference step at which the follow of phase ph hands back (v10p)."""
        src = self.cfg["src"]
        n, phases = len(src["actions"]), src["phases"]
        return n + HOLD if ph >= len(phases) - 1 else min(phases[ph][1] + SETTLE, n + HOLD)

    def _start_seg(self, ph, tau0, d, why):
        self.follow, self.tau0, self.t_start, self.d, self.seg_ph = True, int(tau0), self.t, d, ph
        if getattr(self, "_last", None) is not None:  # the hand offset at the segment start (V10-3 --state phase)
            inner = self.env.unwrapped._env.env
            src = self.cfg["src"]
            eef = np.array(inner.sim.data.site_xpos[inner.robots[0].eef_site_id])
            jr = int(np.argmin(np.abs(src["t"] - int(tau0))))
            self.seg_off0 = eef - src["eef"][jr]
        self.segs.append(dict(t=self.t, ph=int(ph), tau0=int(tau0), why=why))

    def _end_seg(self, inner):
        """Hand back to pi0.5 after following phase seg_ph; set pi0.5's progress tracking to the phase's outcome."""
        src, ph = self.cfg["src"], self.seg_ph
        n, phases = len(src["actions"]), src["phases"]
        self.follow, self.pend, self.close_t, self.last_ev, self.next_check = False, 0, None, self.t, self.t
        if ph >= len(phases) - 1:
            self.finished = True
            self.segs[-1].update(end=self.t)
            return
        closed = src["actions"][min(phases[ph][1], n - 1)][6] > 0  # the phase ends in a grasp (close) or a release
        q = inner.sim.data.qpos[inner.robots[0]._ref_gripper_joint_pos_indexes]
        held = closed and float(q[0] - q[1]) >= EMPTY
        self.cmd = closed
        if closed and not held:  # the followed grasp came up empty: progress stays, pi0.5 (or a takeover) retries
            self.holding, self.k = False, ph
        else:
            self.holding, self.k = held, ph + 1
        self.segs[-1].update(end=self.t, held=bool(held) if closed else None)

    def _decide(self, kind, ph, go_model, p_explore, pf, pb, d):
        """explore: a random choice, logged with the network's inputs at this decision point; v10v: the value head's
        choice; else the model's choice."""
        if self.cfg.get("value") and "vlow" in _CACHE:  # V10-3 with the low-dim logistic value head
            vz = _CACHE["vlow"]
            x = self._x[0].cpu().numpy().astype(np.float64)  # P(follow), P(pi0.5), |d| cm
            kd = {"start": 0, "boundary": 1}.get(kind, 2)
            f = np.array([x[0], x[1], x[0] - x[1], x[2] / 5.0, kd == 0, kd == 1,
                          ph / max(len(self.cfg["src"]["phases"]), 1), float(self._last["tau"]) / 600.0], dtype=np.float64)
            p = [float(1 / (1 + np.exp(-(((f - vz[f"{nm}_mu"]) / vz[f"{nm}_sd"]) @ vz[f"{nm}_w"] + vz[f"{nm}_b"][0]))))
                 for nm in ("cont", "take")]
            go = bool(p[1] - p[0] > float(vz["margin"]))
            self.log.append(dict(t=self.t, decision=kind, ph=int(ph), p_cont=round(p[0], 3), p_take=round(p[1], 3), go=go))
            return go
        if self.cfg.get("value"):
            import train_value as TV

            head, temps = _CACHE["v"]
            with torch.no_grad():
                lg = head(self._h, self._x.cuda(), torch.tensor([TV.kind_id(kind)]).cuda(),
                          torch.tensor([ph / max(len(self.cfg["src"]["phases"]), 1)], dtype=torch.float32).cuda(),
                          torch.tensor([float(self._last["tau"])]).cuda())
                p = torch.sigmoid(lg / temps)[0].cpu().numpy()
            go = bool(p[1] - p[0] > self.cfg["margin_v"])
            self.log.append(dict(t=self.t, decision=kind, ph=int(ph), p_cont=round(float(p[0]), 3), p_take=round(float(p[1]), 3),
                                 go=go))
            return go
        if not self.cfg.get("explore"):
            return go_model
        if self.written:  # after the episode (lerobot autoresets finished envs): nothing to log
            return go_model
        if not hasattr(self, "erng"):
            self.erng = np.random.default_rng(self.cfg["explore_seed"])
            self.dec = []
        go = bool(self.erng.random() < p_explore)
        L = self._last
        self.dec.append(dict(t=self.t, kind=kind, ph=int(ph), go=go, pf=pf, pb=pb, d=np.asarray(d, np.float32), **L))
        z = {k_: np.stack([np.asarray(x[k_]) for x in self.dec]) for k_ in self.dec[0] if k_ != "kind"}
        np.savez(self.cfg["decisions"], kind=np.array([x["kind"] for x in self.dec]), **z)
        return go

    def _takeover_phase(self, a, cont=False, explore=False):
        """v10p: per-phase takeover (see the module docstring). cont (v10q): V10-1's start margin, and at the end of a
        followed phase the heads decide whether to follow the next phase as well."""
        c, src = self.cfg, self.cfg["src"]
        inner = self.env.unwrapped._env.env
        n, phases = len(src["actions"]), src["phases"]
        if self.t == 0 and not getattr(self, "_started", False):
            self._started = True
            d, pf, pb = self._infer(inner, 0, at_start=True)
            self.from_start = bool((pf - pb) > (c["margin"] if cont else c["margin_start"]))
            if explore:
                self.from_start = self._decide("start", 0, self.from_start, 0.5, pf, pb, d)
            self.follow = False
            self.log.append(dict(t=0, tau=0, pf=round(pf, 3), pb=round(pb, 3), d=[round(float(x), 4) for x in d]))
            if self.from_start:
                self._start_seg(0, 0, d, "start")
            self._dump(None)
        if a is None:  # V10-3-2: the decision only (from reset)
            return None
        if self.follow:
            tau = self.tau0 + (self.t - self.t_start)
            if tau >= self._seg_end(self.seg_ph):
                ph_next = self.seg_ph + 1
                if cont and ph_next < len(phases):  # V10-2b: keep following the next phase?
                    tau1 = phases[ph_next][0]
                    d1, pf, pb = self._infer(inner, tau1, at_start=True)
                    go = (pf - pb) > c["margin"]
                    if explore:
                        go = self._decide("boundary", ph_next, go, 0.5, pf, pb, d1)
                    self.log.append(dict(t=self.t, tau=int(tau1), boundary=int(ph_next), pf=round(pf, 3), pb=round(pb, 3),
                                         go=bool(go), d=[round(float(x), 4) for x in d1]))
                    if go:
                        self.segs[-1].update(end=self.t)
                        self._start_seg(ph_next, tau, d1, "continue")  # seamless: as V10-1 when always continuing
                        self._dump(None)
                        return self._soft_action(a) if self.soft is not None else self._follow_action(inner, int(tau))
                self._end_seg(inner)
                self._dump(None)
                return a
            if self.soft is not None:  # V10-3-2: pi0.5's own (guided) action; d is re-estimated in _soft_write
                return self._soft_action(a)
            if (self.t - self.t_start) % CALL == 0 and tau < n and self.t > self.t_start:
                self.d = self._infer(inner, tau)[0]
                self.log.append(dict(t=self.t, tau=int(tau), d=[round(float(x), 4) for x in self.d]))
            return self._follow_action(inner, tau)
        if self.finished or self.n_take >= MAX_TAKE:
            return a
        if is_v3(c["mode"]):
            # V10-3 v1: no takeovers while pi0.5 is in control -- the explore data (10-25) holds decisions at step 0 and
            # at the ends of followed phases only (its event / trigger decisions were patched into the unused v10t
            # copy of the tracking code), and V10-2 showed late takeovers rarely help (8 / 29)
            return a
        closed = a[6] > 0  # pi0.5 in control: its grasp / release progress from its own gripper commands
        self.pend = self.pend + 1 if closed != self.cmd else 0
        if self.pend >= 3:
            self.cmd, self.pend = closed, 0
            if closed:
                self.close_t = self.t
            elif self.holding:
                self.holding, self.k, self.last_ev = False, self.k + 1, self.t  # a release
        trig = None
        if self.cmd and self.close_t is not None and self.t - self.close_t == GRASP_CHECK:
            q = inner.sim.data.qpos[inner.robots[0]._ref_gripper_joint_pos_indexes]
            if float(q[0] - q[1]) >= EMPTY:
                self.holding, self.k, self.last_ev = True, self.k + 1, self.t  # a grasp
            else:
                trig = "empty"
        ph = min(self.k, len(phases) - 1)
        length = max(phases[ph][1] - phases[ph][0], CALL)
        if trig is None and self.t >= self.next_check and self.t - self.last_ev > STUCK_X * length + STUCK_ADD:
            trig = "stuck"
        if trig is None:
            return a
        self.next_check = self.t + RECHECK
        tau0 = phases[ph][0]
        d, pf, pb = self._infer(inner, tau0)
        go = (pf - pb) > c["margin"]
        self.triggers.append(dict(t=self.t, why=trig, k=self.k, tau0=int(tau0), pf=round(pf, 3), pb=round(pb, 3),
                                  go=bool(go)))
        self._dump(None)
        if not go:
            return a
        self.n_take += 1
        self.taken = True
        self._start_seg(ph, tau0, d, trig)
        return self._follow_action(inner, int(tau0))


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
    if cfg.get("aug") is not None:  # training scenes: pi0.5 sees the scene augmentation too
        import augment as A

        return V10Env(A.AugEnv(func(k, **kw), cfg["aug"]), cfg)
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


def orig_entries(args, tower, mean, comp, InstructionText):
    """Gate 2 (10-27; registered 10-22, desktop design doc 6.3): the 10 ORIGINAL libero_10 tasks, unperturbed, init
    states --inits (default 0..5: never used by training, validation or the value head's exploration, all of which
    used 6..49). References: 3 pi0.5 successes of the same task at OTHER init states (runs/v10b/refs), drawn per task
    with a fixed seed and shared by the task's cells (as the 3 sources per base task of the camera test), so an exact
    copy is impossible; replayed and encoded without scene augmentation, like the test scenes. Env seed
    1000 + 7 * task + init as in the camera test; base = two seeds of pi0.5's sampling."""
    import collections
    import pickle

    V = f"{ROOT}/v10b"
    rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
    recs = pickle.load(open(f"{V}/pairs/refs_rec.pkl", "rb"))
    lang = {r["task_id"]: r["language"] for r in rows}
    pools = collections.defaultdict(list)
    for r in rows:
        rec = recs.get((r["task_id"], r["init"]), {})
        if not (r["success"] and rec.get("success")) or r["init"] in args.inits:
            continue
        acts = np.load(f"{V}/refs/{r['actions']}").astype(np.float64)
        n = len(acts)
        pools[r["task_id"]].append(dict(
            key=(r["task_id"], r["init"]), task_id=int(r["task_id"]), init_index=int(r["init"]), seed=int(r["seed"]),
            language=r["language"], actions=acts, phases=[(int(a), int(b)) for a, b, _ in O.phase_targets(rec)],
            path=np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)])))
    chosen = {}
    for tid, pool in sorted(pools.items()):
        rng = np.random.default_rng([tid, 27])
        chosen[tid] = [pool[i] for i in sorted(rng.choice(len(pool), min(3, len(pool)), replace=False))]
        print(f"[orig] task {tid}: {len(pool)} candidate references, using inits "
              f"{[s['init_index'] for s in chosen[tid]]}", flush=True)
    prepare_refs([s for v in chosen.values() for s in v], args.out, tower, mean, comp)
    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(dict.fromkeys(lang.values())))
    entries = []
    for tid in sorted(lang):
        for k in args.inits:
            seed = 1000 + 7 * tid + k
            for mode in ("v10", "v10v") + tuple(m for m in args.modes if m.startswith("v10s")):
                if mode in args.modes:
                    for s in chosen.get(tid, []):
                        entries.append(dict(mode=mode, task_id=tid, init=k, seed=seed, language=lang[tid], src=s,
                                            key=f"{mode}_t{tid}_i{k}_s{s['task_id']}_i{s['init_index']}"))
            if "base" in args.modes:
                for sd in (seed, seed + 4000):
                    entries.append(dict(mode="base", task_id=tid, init=k, seed=sd, language=lang[tid], src=None,
                                        key=f"base_t{tid}_i{k}_seed{sd}"))
    return entries, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{ROOT}/v10b/train/run1/best.pt")
    ap.add_argument("--pca", default=f"{ROOT}/v10b/feats/pca.npz")
    ap.add_argument("--out", default=f"{ROOT}/v10b/closed/run1")
    ap.add_argument("--modes", nargs="+", default=["v10", "base"])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--targets", default=None, help="JSON of variants (select_noncam.py) instead of the 20 camera ones")
    ap.add_argument("--margin_start", type=float, default=0.3, help="v10t: follow from step 0 only above this")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--train_scenes", action="store_true", help="explore: the original tasks' training scenes")
    ap.add_argument("--value", default=None, help="v10v: the value head (train_value.py)")
    ap.add_argument("--margin_v", type=float, default=0.0, help="v10v: take over iff P(take) - P(cont) > margin_v")
    ap.add_argument("--orig", action="store_true", help="gate 2: the original libero_10 tasks (orig_entries)")
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5], help="--orig: test init states")
    ap.add_argument("--cells", nargs="+", default=None, help="only these cells, task:init (smoke tests)")
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
    if args.targets:  # other categories: base task = the variant's stem, pi0.5 gets the variant's own instruction
        cams = sorted((x["task_id"], dict(basename=x["base"], base=x["language"], category=x["category"]))
                      for x in json.load(open(args.targets)))

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

    if args.train_scenes:
        return main_train_scenes(args, policy, noise, tower_path, G, InstructionText, LiberoPlusEnv,
                                 make_env_pre_post_processors, _LazyAsyncVectorEnv, make_pre_post_processors, rollout,
                                 FixedTasks)
    if args.orig:
        entries, text = orig_entries(args, tower, mean, comp, InstructionText)
        del tower
        torch.cuda.empty_cache()
    else:
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
                for mode in ("v10", "v10t", "v10p", "v10q", "v10v") + tuple(m for m in args.modes if m.startswith("v10s")):
                    if mode in args.modes:
                        for s in pool:
                            entries.append(dict(mode=mode, task_id=tid, init=k, seed=seed, language=r["base"], src=s,
                                                key=f"{mode}_t{tid}_i{k}_s{s['task_id']}"))
                if "base" in args.modes:
                    for sd in (seed, seed + 4000):
                        entries.append(dict(mode="base", task_id=tid, init=k, seed=sd, language=r["base"], src=None,
                                            key=f"base_t{tid}_i{k}_seed{sd}"))
    if args.cells:
        keep = {tuple(int(v) for v in x.split(":")) for x in args.cells}
        entries = [e for e in entries if (e["task_id"], e["init"]) in keep]
    if args.check_noise:  # the same 8 base episodes in two batch orders
        base = [e for e in entries if e["mode"] == "base"][:16:2]
        entries = [dict(e, key=e["key"] + "_o1") for e in base] + [dict(e, key=e["key"] + "_o2") for e in base[::-1]]
    log_path = f"{args.out}/episodes.jsonl"
    done = {json.loads(l)["key"] for l in open(log_path)} if os.path.exists(log_path) else set()
    todo = [e for e in entries if e["key"] not in done]
    if args.limit:  # smoke test: the first episodes only
        todo = todo[:args.limit]
    print(f"[closed] {len(entries)} episodes ({len(todo)} to do): "
          + ", ".join(f"{m} {sum(e['mode'] == m for e in entries)}" for m in args.modes), flush=True)

    env_cfg = LiberoPlusEnv(task="libero_10")
    pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                         preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                                 "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    soft = install_soft(post) if any(m.startswith("v10s") for m in args.modes) else None
    log = open(log_path, "a")
    t0 = time.time()
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        fns = []
        for i_row, e in enumerate(batch):
            fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=e["task_id"], n_envs=1,
                               camera_names=G["cams"], episode_length=env_cfg.episode_length,
                               init_states=env_cfg.init_states, gym_kwargs=gk, control_mode=env_cfg.control_mode,
                               camera_name_mapping=env_cfg.camera_name_mapping, is_libero_plus=env_cfg.is_libero_plus)[0]
            cfg = dict(mode=e["mode"], margin=args.margin, margin_start=args.margin_start, ckpt=args.ckpt, pca=args.pca,
                       value=args.value if is_v3(e["mode"]) else None, margin_v=args.margin_v,
                       tower=tower_path, key=e["key"], soft_path=f"{SOFT_DIR}/{i_row}.npz",
                       log=f"{args.out}/wrapper/{e['key']}.json")
            if e["src"] is not None:
                s = e["src"]
                cfg["src"] = {k: s[k] for k in ("actions", "path", "phases", "feats", "t", "eef", "grip")}
                cfg["lang"] = [text.embed(e["language"]).numpy(), text.embed(s["language"]).numpy()]
            fns.append(functools.partial(v10_env, fn.func, e["init"], cfg, **fn.keywords))
        noise.set([e["seed"] for e in batch])
        if soft is not None:
            soft.new_batch([e["key"] for e in batch], [f"{SOFT_DIR}/{i}.npz" for i in range(len(batch))])
        env = _LazyAsyncVectorEnv(fns)
        try:
            torch.manual_seed(batch[0]["seed"])
            ret = rollout(env=env, policy=policy, env_preprocessor=FixedTasks(env_pre, [e["language"] for e in batch]),
                          env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                          seeds=[e["seed"] for e in batch])
        finally:
            env.close()
        if soft is not None:
            print(f"[soft] calls guided {soft.stats['guided']}, stale {soft.stats['stale']}", flush=True)
        succ = ret["success"].bool().any(dim=1).tolist()
        dn = ret["done"].bool()
        for i, e in enumerate(batch):
            T = int(torch.argmax(dn[i].int())) + 1 if bool(dn[i].any()) else dn.shape[1]
            w = f"{args.out}/wrapper/{e['key']}.json"
            extra = json.load(open(w)) if os.path.exists(w) else {}
            log.write(json.dumps(dict(key=e["key"], mode=e["mode"], task_id=e["task_id"], init=e["init"], seed=e["seed"],
                                      source=e["src"]["task_id"] if e["src"] else None,
                                      source_init=e["src"].get("init_index") if e["src"] else None,
                                      success=bool(succ[i]), steps=T,
                                      follow=extra.get("follow"), taken=extra.get("taken"),
                                      from_start=extra.get("from_start"), gpu=torch.cuda.get_device_name(0))) + "\n")
        log.flush()
        print(f"[closed] {b0 + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s), batch success {sum(succ)}/{len(batch)}",
              flush=True)
    print(f"-> {args.out}", flush=True)


def main_train_scenes(args, policy, noise, tower_path, G, InstructionText, LiberoPlusEnv, make_env_pre_post_processors,
                      _LazyAsyncVectorEnv, make_pre_post_processors, rollout, FixedTasks):
    """explore on the training scenes (see the module docstring)."""
    import pickle

    import augment as A
    import collect_pi05 as CP

    V = f"{ROOT}/v10b"
    refs_rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
    recs = pickle.load(open(f"{V}/pairs/refs_rec.pkl", "rb"))
    lang = {r["task_id"]: r["language"] for r in refs_rows}
    srcs = {}
    for r in refs_rows:
        key = f"t{r['task_id']}_i{r['init']}"
        if not (r["success"] and recs.get((r["task_id"], r["init"]), {}).get("success")
                and os.path.exists(f"{V}/feats/refs/{key}_a0.npy")):
            continue
        rec = recs[(r["task_id"], r["init"])]
        acts = np.load(f"{V}/refs/{r['actions']}").astype(np.float64)
        n = len(acts)
        fm = np.load(f"{V}/frames/refs/{key}_a0.npz")
        srcs[key] = dict(task_id=r["task_id"], init=r["init"], language=r["language"], actions=acts,
                         phases=[(int(a), int(b)) for a, b, _ in O.phase_targets(rec)],
                         path=np.concatenate([rec["path"], np.repeat(rec["path"][-1:], max(0, n - len(rec["path"])), 0)]),
                         feats=np.load(f"{V}/feats/refs/{key}_a0.npy"), t=fm["t"], eef=fm["eef_pos"], grip=fm["gripper"])
    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(dict.fromkeys(lang.values())))
    VAL = set(range(45, 50))
    entries = []
    for tid in sorted(lang):
        for k in range(6, 50):
            rng = np.random.default_rng([tid, k, 11])
            pool = [s_ for s_ in srcs.values() if s_["task_id"] == tid and s_["init"] != k and ((s_["init"] in VAL) == (k in VAL))]
            if not pool:
                continue
            src = pool[int(rng.integers(len(pool)))]
            entries.append(dict(mode="explore", task_id=tid, init=k, seed=3000 + 100 * (tid - CP.OFFSET) + k,
                                language=lang[tid], src=src, key=f"explore_t{tid}_i{k}_t{src['task_id']}_i{src['init']}",
                                split="val" if k in VAL else "train"))
    os.makedirs(f"{args.out}/decisions", exist_ok=True)
    log_path = f"{args.out}/episodes.jsonl"
    done = {json.loads(l)["key"] for l in open(log_path)} if os.path.exists(log_path) else set()
    todo = [e for e in entries if e["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[closed] {len(entries)} episodes ({len(todo)} to do): explore on training scenes", flush=True)
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
            s_ = e["src"]
            cfg = dict(mode="explore", margin=args.margin, margin_start=args.margin_start, ckpt=args.ckpt, pca=args.pca,
                       tower=tower_path, log=f"{args.out}/wrapper/{e['key']}.json", explore=True,
                       explore_seed=int(e["seed"]) * 7 + 1, decisions=f"{args.out}/decisions/{e['key']}.npz",
                       aug=A.scene_aug(e["task_id"], e["init"], 0),
                       src={k_: s_[k_] for k_ in ("actions", "path", "phases", "feats", "t", "eef", "grip")},
                       lang=[text.embed(e["language"]).numpy(), text.embed(s_["language"]).numpy()])
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
        for i, e in enumerate(batch):
            log.write(json.dumps(dict(key=e["key"], mode="explore", task_id=e["task_id"], init=e["init"], seed=e["seed"],
                                      ref_task=e["src"]["task_id"], ref_init=e["src"]["init"], split=e["split"],
                                      success=bool(succ[i]), gpu=torch.cuda.get_device_name(0))) + "\n")
        log.flush()
        print(f"[closed] {b0 + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s), batch success {sum(succ)}/{len(batch)}",
              flush=True)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
