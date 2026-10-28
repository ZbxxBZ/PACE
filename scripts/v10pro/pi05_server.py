"""
pi0.5 policy server for the LIBERO-PRO clients (10-23). Runs in the LIBERO-plus / lerobot env (.venv_plus) on the GPU.
A client sends the simulator's raw observation fields (both 360 px cameras, eef pos / quat / controller orientation
matrix, gripper and joint states), the task's language, the episode seed and the call index; the server rebuilds the
exact observation dict lerobot's LiberoEnv._format_raw_obs makes, runs the closed loop's own preprocessing
(preprocess_observation -> task -> LIBERO env preprocessor -> policy preprocessor), predicts a chunk with noise drawn
from a generator seeded by (seed, call) so every episode is reproducible, keeps the first n_action_steps (10) and
returns them through the policy postprocessor and the env postprocessor: env-unit actions [10, 7]. One inference at
a time (a lock); clients run their simulators in parallel.
    python pi05_server.py --port 18765
"""
import argparse
import socketserver
import sys
import threading
import time

import numpy as np
import torch

from pro_wire import recv_msg, send_msg

STATE = {}


def setup(policy_path):
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from pace.common import load_policy

    policy, preprocessor = load_policy(policy_path, "cuda", n_action_steps=10)
    _, postprocessor = make_pre_post_processors(policy.config, pretrained_path=policy_path,
                                                preprocessor_overrides={"device_processor": {"device": "cuda"}})
    env_pre, env_post = make_env_pre_post_processors(LiberoEnv(task="libero_10"), policy.config)
    STATE.update(policy=policy, pre=preprocessor, post=postprocessor, env_pre=env_pre, env_post=env_post,
                 lock=threading.Lock(), calls=0)


@torch.no_grad()
def infer(h, a):
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION

    b = lambda x: np.asarray(x)[None]  # the vector env's batch dimension
    obs = {"pixels": {"image": b(a["agentview_image"]), "image2": b(a["robot0_eye_in_hand_image"])},
           "robot_state": {"eef": {"pos": b(a["eef_pos"]), "quat": b(a["eef_quat"]), "mat": b(a["eef_mat"])},
                           "gripper": {"qpos": b(a["gripper_qpos"]), "qvel": b(a["gripper_qvel"])},
                           "joints": {"pos": b(a["joint_pos"]), "vel": b(a["joint_vel"])}}}
    o = preprocess_observation(obs)
    o["task"] = [h["task"]]
    batch = STATE["pre"](STATE["env_pre"](o))
    p = STATE["policy"]
    g = torch.Generator(device="cuda").manual_seed(int(h["seed"]) * 100003 + int(h["call"]))
    noise = torch.randn((1, p.config.chunk_size, p.config.max_action_dim), generator=g, device="cuda",
                        dtype=torch.float32)
    chunk = p.predict_action_chunk(batch, noise=noise)[:, :p.config.n_action_steps]  # [1, 10, A]
    acts = STATE["post"](chunk[0])
    out = STATE["env_post"]({ACTION: acts})[ACTION]
    out = out if torch.is_tensor(out) else torch.as_tensor(np.asarray(out))
    return out.float().cpu().numpy()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                h, a = recv_msg(self.request)
            except ConnectionError:
                return
            if h.get("op") == "ping":
                send_msg(self.request, {"ok": True})
                continue
            with STATE["lock"]:
                t0 = time.time()
                acts = infer(h, a)
                STATE["calls"] += 1
            send_msg(self.request, {"ok": True, "ms": round(1000 * (time.time() - t0), 1)}, {"actions": acts})


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18765)
    ap.add_argument("--policy", default="lerobot/pi05_libero_finetuned")
    args = ap.parse_args()
    setup(args.policy)
    print(f"[server] {args.policy} on cuda, listening on 127.0.0.1:{args.port}", flush=True)
    with Server(("127.0.0.1", args.port), Handler) as srv:
        srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
