"""
LIBERO-PRO side helpers (Python 3.8, LIBERO-PRO's own code; 10-23): building a task's env, the reset lerobot's
LiberoEnv does (seed -> reset -> set_init_state -> 10 no-op settle steps, gripper open), the raw observation fields the
pi0.5 server needs, and a client for that server.
"""
import os
import socket

import numpy as np

from pro_wire import recv_msg, send_msg

SIZE = 360            # lerobot's LIBERO camera size
SETTLE = 10           # lerobot's num_steps_wait
MAX_STEPS = {"libero_10": 520, "libero_spatial": 220, "libero_object": 280, "libero_goal": 300}
NOOP = [0, 0, 0, 0, 0, 0, -1]


def suite_of(name):
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[name]()


def max_steps(suite_name):
    return MAX_STEPS[next(k for k in MAX_STEPS if suite_name.startswith(k))]


def make_env(suite_name, task_idx, size=SIZE):
    """(env, task, init states) of one LIBERO-PRO task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = suite_of(suite_name)
    task = suite.get_task(task_idx)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size, control_freq=20)
    env.reset()
    return env, task, suite.get_task_init_states(task_idx)


def reset(env, init_state, seed):
    env.seed(int(seed))
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(SETTLE):
        obs, _, _, _ = env.step(NOOP)
    return obs


def raw_fields(env, obs):
    ctrl = env.env.robots[0].controller
    return {"agentview_image": obs["agentview_image"], "robot0_eye_in_hand_image": obs["robot0_eye_in_hand_image"],
            "eef_pos": obs["robot0_eef_pos"], "eef_quat": obs["robot0_eef_quat"], "eef_mat": np.asarray(ctrl.ee_ori_mat),
            "gripper_qpos": obs["robot0_gripper_qpos"], "gripper_qvel": obs["robot0_gripper_qvel"],
            "joint_pos": obs["robot0_joint_pos"], "joint_vel": obs["robot0_joint_vel"]}


class PolicyClient:
    def __init__(self, port=18765):
        self.sock = socket.create_connection(("127.0.0.1", port))

    def act(self, env, obs, task, seed, call):
        send_msg(self.sock, {"task": task, "seed": int(seed), "call": int(call)}, raw_fields(env, obs))
        h, a = recv_msg(self.sock)
        return a["actions"]

    def close(self):
        self.sock.close()
