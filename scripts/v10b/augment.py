"""
Generic visual augmentation for V10-1b training (10-24; docs/PACE_V1-V8.md "数据来源": our own ranges, no LIBERO-plus
variant files). Only fields MuJoCo uses for rendering are changed (camera pose, lights, material colours) plus pixel
noise / blur on the returned images, so the physics -- and therefore the replay of stored actions -- is unchanged
(checked by render_pairs.py --check).
  camera   : agentview position +-CAM_POS m per axis, rotation about a random axis by up to CAM_ROT deg (the wrist
             camera is mounted on the arm and is left alone);
  light    : diffuse intensity x U(LIGHT), per-channel tint x U(TINT), direction jitter;
  surfaces : per-channel colour x U(SURF) for the materials of the table, floor and walls;
  pixels   : Gaussian noise (sigma up to NOISE, 0-255 units) with prob. 0.5, box blur 3 / 5 with prob. 0.3.
sample(rng) -> params (a dict of plain numbers, stored next to the rendered frames); apply(inner, params) on a reset
robosuite env; pixels(img, params, rng) per image.
"""
import numpy as np

CAM_POS, CAM_ROT = 0.10, 12.0
LIGHT, TINT, SURF = (0.5, 1.5), (0.8, 1.2), (0.6, 1.4)
NOISE = 8.0
SURF_KEYS = ("table", "floor", "wall")


def sample(rng):
    ax = rng.normal(size=3)
    return dict(cam_dpos=rng.uniform(-CAM_POS, CAM_POS, 3).tolist(), cam_axis=(ax / np.linalg.norm(ax)).tolist(),
                cam_angle=float(rng.uniform(0, CAM_ROT)), light=float(rng.uniform(*LIGHT)),
                tint=rng.uniform(*TINT, 3).tolist(), light_dir=rng.normal(0, 0.15, 3).tolist(),
                surf=rng.uniform(*SURF, 3).tolist(), noise=float(rng.uniform(0, NOISE)) if rng.random() < 0.5 else 0.0,
                blur=int(rng.choice([3, 5])) if rng.random() < 0.3 else 0, seed=int(rng.integers(1 << 30)))


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def scene_aug(tid, k, s):
    """The s-th augmentation of current scene (task id, init state): shared by that scene's pair renders (pair s) and
    the pi0.5 rollout that labels it (collect_pi05.py --set orig_aug)."""
    import zlib

    return sample(np.random.default_rng([zlib.crc32(f"scene/t{tid}_i{k}".encode()), s]))


FIELDS = ("cam_pos", "cam_quat", "light_diffuse", "light_dir", "mat_rgba")


def apply(inner, p):
    """Change the rendering-only model fields of a reset robosuite env in place (from the original values, so applying
    again after a reset that kept the model does not compound)."""
    m = inner.sim.model
    orig = getattr(inner, "_aug_orig", None)
    if orig is None or orig[0] is not m:
        inner._aug_orig = orig = (m, {f: getattr(m, f).copy() for f in FIELDS})
    for f, v in orig[1].items():
        getattr(m, f)[:] = v
    cid = m.camera_name2id("agentview")
    m.cam_pos[cid] = m.cam_pos[cid] + np.array(p["cam_dpos"])
    h = np.deg2rad(p["cam_angle"]) / 2
    m.cam_quat[cid] = _qmul(np.r_[np.cos(h), np.sin(h) * np.array(p["cam_axis"])], m.cam_quat[cid])
    for i in range(m.nlight):
        m.light_diffuse[i] = np.clip(m.light_diffuse[i] * p["light"] * np.array(p["tint"]), 0, 1)
        d = m.light_dir[i] + np.array(p["light_dir"])
        m.light_dir[i] = d / max(np.linalg.norm(d), 1e-6)
    import mujoco

    raw = getattr(m, "_model", m)
    for i in range(m.nmat):
        name = (mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_MATERIAL, i) or "").lower()
        if any(k in name for k in SURF_KEYS):
            m.mat_rgba[i, :3] = np.clip(m.mat_rgba[i, :3] * np.array(p["surf"]), 0, 1)
    inner.sim.forward()


def pixels(img, p, rng):
    """uint8 HxWx3 -> uint8, noise and blur per the params (rng: per-frame noise draws)."""
    x = img.astype(np.float32)
    if p["blur"]:
        k = p["blur"]
        pad = np.pad(x, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="edge")
        c = np.cumsum(np.cumsum(pad, 0), 1)
        c = np.pad(c, ((1, 0), (1, 0), (0, 0)))
        x = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    if p["noise"]:
        x = x + rng.normal(0, p["noise"], x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def aug_env(func, k, aug, **kw):
    """Factory for lerobot's vector env: the LiberoEnv that func(k, **kw) builds, augmented."""
    return AugEnv(func(k, **kw), aug)


def _wrapper_base():
    import gymnasium as gym

    return gym.Wrapper


class AugEnv(_wrapper_base()):
    """Applies `aug` after every reset and to every returned image (the policy sees what the renders store)."""

    def __init__(self, env, aug):
        super().__init__(env)
        self.aug = aug
        self.rng = np.random.default_rng(aug["seed"])

    def _pix(self, obs):
        if isinstance(obs, dict) and "pixels" in obs:
            obs = dict(obs)
            obs["pixels"] = {c: pixels(np.ascontiguousarray(v), self.aug, self.rng) for c, v in obs["pixels"].items()}
        return obs

    def reset(self, **kw):
        _, info = self.env.reset(**kw)
        inner = self.env.unwrapped._env.env
        apply(inner, self.aug)
        obs = self.env.unwrapped._format_raw_obs(inner._get_observations(force_update=True))
        return self._pix(obs), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        return self._pix(obs), r, term, trunc, info
