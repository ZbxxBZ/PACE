"""Env wrapper for v11b_noncam.py: keeps the raw observation at every pi0.5 call of the first episode and writes it when it ends."""
import gymnasium as gym
import numpy as np

CALL = 10


class RecEnv(gym.Wrapper):
    def __init__(self, env, path, agent, wrist):
        super().__init__(env)
        self.path, self.agent, self.wrist = path, agent, wrist
        self.t, self.closed = 0, False
        self.rec = {k: [] for k in ("agent", "wrist", "t", "eef_pos", "eef_quat", "joints", "gripper")}

    def _keep(self, obs):
        px, st = obs["pixels"], obs["robot_state"]
        self.rec["agent"].append(np.array(px[self.agent]))
        self.rec["wrist"].append(np.array(px[self.wrist]))
        self.rec["t"].append(self.t)
        self.rec["eef_pos"].append(np.array(st["eef"]["pos"]))
        self.rec["eef_quat"].append(np.array(st["eef"]["quat"]))
        self.rec["joints"].append(np.array(st["joints"]["pos"]))
        self.rec["gripper"].append(np.array(st["gripper"]["qpos"]))

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        if not self.closed and self.t == 0 and not self.rec["t"]:
            self._keep(obs)
        return obs, info

    def _write(self, success):
        self.closed = True
        np.savez_compressed(self.path, **{k: np.stack(v) for k, v in self.rec.items()}, success=bool(success), steps=self.t)

    def flush(self):
        """An episode that ran out of steps gets no done from the env: the caller writes it after the rollout."""
        if not self.closed:
            self._write(False)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if not self.closed:
            self.t += 1
            if term or trunc:
                self._write(info.get("is_success", False))
            elif self.t % CALL == 0:
                self._keep(obs)
        return obs, r, term, trunc, info


def rec_env(func, k, path, agent, wrist, **kw):
    return RecEnv(func(k, **kw), path, agent, wrist)
