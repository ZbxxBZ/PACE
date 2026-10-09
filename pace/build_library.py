"""
Build a demo experience library (an ExperienceStore file) from a LeRobot dataset with a frozen pi0.5.

Each selected demonstration becomes one successful `demo` episode keyed by its normalized instruction. One entry is
kept per policy call: frames 0, s, 2s, ... of the episode with s = --stride (default: the policy's n_action_steps, as
the evaluation calls the policy every n_action_steps env steps). Per call it stores the retrieval key (see
`pace/pi05.py`), the normalized action chunk of chunk_size steps from that frame on (the policy preprocessor's
normalization, the same chunk the policy is trained to predict there) and, per episode, the first-frame signature.
Only the vision tower runs (no language model), so this is cheap.

    python -m pace.build_library --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
        --suite libero_goal --n_action_steps 10 --out libs/libero_goal_demo.pt
"""

import argparse
import collections
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.utils.constants import ACTION

from .core import ExperienceStore, make_episode, normalize_instruction
from .pi05 import call_keys, key_info
from .common import load_dataset, load_policy, suite_instructions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", default=None, help="Keep only the tasks of this LIBERO suite (e.g. libero_goal)")
    parser.add_argument("--tasks_file", default=None, help="JSON list of instructions to keep")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Override the policy's n_action_steps")
    parser.add_argument("--stride", type=int, default=0, help="Frames per policy call (0 = n_action_steps)")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1], help="Camera indices the key pools")
    parser.add_argument("--sig_grid", type=int, default=4)
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, preprocessor = load_policy(args.policy_path, args.device, n_action_steps=args.n_action_steps)
    stride = args.stride or policy.config.n_action_steps
    dataset = load_dataset(policy, args.dataset_repo_id, args.dataset_root, args.video_backend)

    keep = None
    if args.suite:
        keep = {normalize_instruction(t) for t in suite_instructions(args.suite)}
    if args.tasks_file:
        with open(args.tasks_file) as f:
            listed = {normalize_instruction(t) for t in json.load(f)}
        keep = listed if keep is None else keep & listed

    episodes_meta = dataset.meta.episodes
    starts, ends = list(episodes_meta["dataset_from_index"]), list(episodes_meta["dataset_to_index"])
    all_tasks = list(episodes_meta["tasks"])
    selected = []
    for ep in range(dataset.meta.total_episodes):
        tasks = all_tasks[ep]
        task = tasks[0] if isinstance(tasks, (list, tuple)) else tasks
        if keep is not None and normalize_instruction(task) not in keep:
            continue
        frames = list(range(int(starts[ep]), int(ends[ep]), stride))
        selected.append((ep, task, frames))
        if args.max_episodes and len(selected) >= args.max_episodes:
            break
    if keep is not None:
        found = {normalize_instruction(task) for _, task, _ in selected}
        missing = sorted(keep - found)
        assert not missing or args.max_episodes, f"tasks without episodes in the dataset: {missing}"
    print(f"[library] {len(selected)} episodes, stride {stride}, chunk {policy.config.chunk_size}", flush=True)

    info = key_info(args.cameras, args.sig_grid)
    store = ExperienceStore(info=info)
    loader = DataLoader(dataset, batch_sampler=[frames for _, _, frames in selected], num_workers=args.num_workers,
                        collate_fn=default_collate)
    started = time.time()
    for n, ((ep, task, frames), raw) in enumerate(zip(selected, loader), 1):
        batch = preprocessor(raw)
        keys, sigs = call_keys(policy, batch, args.cameras, args.sig_grid)
        actions = batch[ACTION].float()  # [T, chunk, A] normalized
        assert actions.shape[1] == policy.config.chunk_size, f"action chunk {tuple(actions.shape)}"
        store.add(make_episode(
            normalize_instruction(task), keys, actions, sigs[0], True, "demo", name=f"demo/ep{ep:05d}",
            meta={"episode_index": ep, "dataset_indices": frames, "task": task, "stride": stride,
                  "dataset": args.dataset_repo_id, "policy_path": args.policy_path},
        ))
        if n % 50 == 0:
            print(f"[library] {n}/{len(selected)} episodes ({time.time() - started:.0f}s)", flush=True)

    lengths = collections.defaultdict(list)
    for episode in store.episodes:
        lengths[episode["task_key"]].append(episode["keys"].shape[0])
    for task, values in sorted(lengths.items()):
        print(f"{len(values):4d} demos, {min(values)}-{max(values)} calls: {task}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    store.save(args.out)
    print(f"saved {len(store)} episodes of {len(lengths)} tasks to {args.out} (keys {info})")


if __name__ == "__main__":
    main()
