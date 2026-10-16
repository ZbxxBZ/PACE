"""Loading helpers shared by the PACE scripts (policy, processors, dataset, task lists, prompt overrides)."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .core import normalize_instruction


def load_policy(policy_path: str, device: str = "cuda", n_action_steps: Optional[int] = None):
    """
    A frozen, eval-mode PI05Policy and its preprocessor (normalization, prompt tokenization, device). The config is
    overridden before the model is built: compile_model off (lerobot/pi05_libero_finetuned stores True, and a compiled
    sample_actions would bypass PACE's hooks) and n_action_steps (that checkpoint stores 50; LIBERO uses 10).
    """
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(policy_path)
    config.compile_model = False
    if n_action_steps:
        config.n_action_steps = int(n_action_steps)
    config.device = device
    policy = PI05Policy.from_pretrained(policy_path, config=config)
    check_loaded(policy, policy_path)
    policy.to(device)
    policy.eval()
    policy.requires_grad_(False)
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=policy_path, preprocessor_overrides={"device_processor": {"device": device}}
    )
    return policy, preprocessor


def check_loaded(policy, policy_path: str) -> None:
    """
    PI05Policy.from_pretrained returns a randomly initialized model (with only a printed note) when it cannot read the
    weights; compare the action output projection with the checkpoint file instead of trusting it.
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = Path(policy_path) / "model.safetensors"
    if not path.exists():
        path = Path(hf_hub_download(policy_path, "model.safetensors"))
    with safe_open(str(path), framework="pt") as f:
        names = [k for k in f.keys() if k.endswith("action_out_proj.weight")]
        assert len(names) == 1, f"{path}: action_out_proj.weight keys {names}"
        stored = f.get_tensor(names[0]).float()
    loaded = policy.model.action_out_proj.weight.detach().float().cpu()
    if stored.shape != loaded.shape or not torch.allclose(stored, loaded, atol=1e-2, rtol=1e-2):
        raise RuntimeError(f"{policy_path}: the policy's weights do not match {path} (load failed?)")


def parse_camera_map(items: Optional[List[str]]) -> Dict[str, str]:
    """["observation.images.front=observation.images.image", ...] -> {dataset key: policy key}."""
    mapping = {}
    for item in items or []:
        src, sep, dst = item.partition("=")
        assert sep and src and dst, f"--camera_map entries are <dataset key>=<policy key>, got {item!r}"
        mapping[src] = dst
    return mapping


class RenamedDataset(torch.utils.data.Dataset):
    """A dataset whose camera keys (and their `_is_pad` companions) are renamed to the policy's."""

    def __init__(self, inner, mapping: Dict[str, str]) -> None:
        self.inner, self.mapping = inner, dict(mapping)

    def rename(self, key: str) -> str:
        for src, dst in self.mapping.items():
            if key == src or key.startswith(src + "_is_pad"):
                return dst + key[len(src):]
        return key

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, index):
        return {self.rename(k): v for k, v in self.inner[index].items()}

    def __getitems__(self, indices):
        # DataLoader fetches whole batches through __getitems__ when the dataset has one (LeRobotDataset does);
        # without this override __getattr__ would hand it the inner one and the keys would stay unrenamed
        getitems = getattr(self.inner, "__getitems__", None)
        items = getitems(list(indices)) if getitems is not None else [self.inner[i] for i in indices]
        return [{self.rename(k): v for k, v in item.items()} for item in items]

    def __getattr__(self, name):
        if name == "inner":  # unpickling (DataLoader workers) looks attributes up before __dict__ is restored
            raise AttributeError(name)
        return getattr(self.inner, name)


def load_dataset(policy, repo_id: str, root: Optional[str] = None, video_backend: Optional[str] = None,
                 camera_map: Optional[Dict[str, str]] = None):
    """
    The LeRobot dataset with the action chunks the policy is trained on (chunk_size steps per frame). `camera_map`
    renames the dataset's camera keys to the policy's (lerobot/libero_plus: front / wrist -> image / image2); every
    camera the policy reads must then exist, since pi0.5 would otherwise fill a missing one with a blank image.
    """
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta_timestamps = resolve_delta_timestamps(policy.config, meta)
    dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps, video_backend=video_backend)
    camera_map = dict(camera_map or {})
    unknown = sorted(set(camera_map) - set(meta.camera_keys))
    assert not unknown, f"--camera_map keys not in {repo_id}: {unknown} (its cameras: {meta.camera_keys})"
    available = {camera_map.get(k, k) for k in meta.camera_keys}
    wanted = {k for k in policy.config.image_features if ".empty_camera_" not in k}  # config.empty_cameras: blank
    missing = sorted(wanted - available)
    assert not missing, (f"the policy reads cameras {missing} that {repo_id} lacks (its cameras: {meta.camera_keys}, "
                         f"after --camera_map {camera_map}); map them with --camera_map <dataset key>=<policy key>")
    return RenamedDataset(dataset, camera_map) if camera_map else dataset


def episode_tasks(dataset) -> List[str]:
    """
    Instruction of every episode. Some v3 datasets (e.g. lerobot/libero) keep no per-episode `tasks` column in
    meta/episodes; the frame table's task_index of each episode's first frame is used then (two parquet columns,
    no video decoding).
    """
    import pandas as pd

    meta = dataset.meta
    if "tasks" in meta.episodes.column_names:
        return [t[0] if isinstance(t, (list, tuple)) else t for t in meta.episodes["tasks"]]
    index_to_task = {int(i): t for t, i in meta.tasks["task_index"].items()}
    files = sorted(Path(meta.root, "data").glob("**/*.parquet"))
    assert files, f"no data parquet files under {meta.root}/data"
    frames = pd.concat([pd.read_parquet(f, columns=["episode_index", "task_index"]) for f in files])
    first = frames.groupby("episode_index")["task_index"].first()
    return [index_to_task[int(first[ep])] for ep in range(meta.total_episodes)]


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
