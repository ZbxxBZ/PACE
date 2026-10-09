"""Loading helpers shared by the PACE scripts (policy, processors, dataset, task lists, prompt overrides)."""

import json
from typing import Dict, List, Optional

from .core import normalize_instruction


def load_policy(policy_path: str, device: str = "cuda", n_action_steps: Optional[int] = None):
    """A frozen, eval-mode PI05Policy and its preprocessor (normalization, prompt tokenization, device)."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(policy_path)
    if n_action_steps:
        policy.config.n_action_steps = int(n_action_steps)
    policy.config.device = device
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=policy_path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, preprocessor


def load_dataset(policy, repo_id: str, root: Optional[str] = None, video_backend: Optional[str] = None):
    """The LeRobot dataset with the action chunks the policy is trained on (chunk_size steps per frame)."""
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, video_backend=video_backend)


def suite_instructions(suite: str) -> List[str]:
    """Instructions of a LIBERO suite, from the installed LIBERO benchmark."""
    from libero.libero import benchmark

    task_suite = benchmark.get_benchmark_dict()[suite]()
    return [task_suite.get_task(i).language for i in range(task_suite.n_tasks)]


class PromptMap:
    """
    What the policy is told instead of the true instruction (hidden-goal experiments). `override` replaces every
    instruction; `mapping` (JSON {instruction: prompt}, "*" = default) replaces the listed ones. The task key used for
    retrieval always stays the true instruction.
    """

    def __init__(self, override: Optional[str] = None, map_file: Optional[str] = None) -> None:
        self.mapping: Dict[str, str] = {}
        if map_file:
            with open(map_file) as f:
                self.mapping = {(k if k == "*" else normalize_instruction(k)): v for k, v in json.load(f).items()}
        if override:
            self.mapping["*"] = override

    def __bool__(self) -> bool:
        return bool(self.mapping)

    def __call__(self, instruction: str) -> str:
        return self.mapping.get(normalize_instruction(instruction), self.mapping.get("*", instruction))
