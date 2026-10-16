"""
Build a demo experience library (an ExperienceStore file) from a LeRobot dataset with a frozen pi0.5.

Each selected demonstration becomes one successful `demo` episode keyed by its normalized instruction. One entry is
kept per policy call: frames 0, s, 2s, ... of the episode with s = --stride (default: the policy's n_action_steps, as
the evaluation calls the policy every n_action_steps env steps). Per call it stores the retrieval key (see
`pace/pi05.py`), the normalized action chunk of chunk_size steps from that frame on (the policy preprocessor's
normalization, the same chunk the policy is trained to predict there) and, per episode, the first-frame signature.
Only the vision tower runs (no language model), so this is cheap. The key transform (--key_transform, default PCA to
--pca_dim after centering) is fitted on the raw keys of all calls (or of --fit_episodes episodes) and stored in the
library, and every stored key is the transformed one; deployment transforms its keys with the same fit. Every call also keeps
its normalized robot state (8 floats; what the v3 retrieval aligns on). With --features 1 every call also keeps what an
adapter with the v2 encoder reads: its image tokens (every camera's 4 x 4 grid of unit cells, fp16, [16 * cameras, D]);
about 128 KB per call with two cameras (all LIBERO demos: ~3.5 GB). The keys are the same with or without features.

    python -m pace.build_library --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
        --suite libero_goal --n_action_steps 10 --out libs/libero_goal_demo.pt
    # lerobot/libero_plus (other camera names; all 14347 demos: ~30 GB of image tokens, loaded memory-mapped)
    python -m pace.build_library --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero_plus         --camera_map observation.images.front=observation.images.image observation.images.wrist=observation.images.image2         --n_action_steps 10 --features 1 --fit_episodes 2000 --out libs/libero_plus_feat.pt
"""

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.utils.constants import ACTION

from .core import ExperienceStore, apply_key_transform, fit_key_transform, make_episode, normalize_instruction
from .pi05 import KEY_KINDS, batch_state, call_keys, key_info
from .common import episode_tasks, load_dataset, load_policy, parse_camera_map, suite_instructions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--camera_map", nargs="*", default=[],
                        help="<dataset key>=<policy key> renames (lerobot/libero_plus: observation.images.front="
                             "observation.images.image observation.images.wrist=observation.images.image2)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--suite", default=None, help="Keep only the tasks of this LIBERO suite (e.g. libero_goal)")
    parser.add_argument("--tasks_file", default=None, help="JSON list of instructions to keep")
    parser.add_argument("--n_action_steps", type=int, default=None, help="Override the policy's n_action_steps")
    parser.add_argument("--stride", type=int, default=0, help="Frames per policy call (0 = n_action_steps)")
    parser.add_argument("--cameras", type=int, nargs="+", default=[0, 1], help="Camera indices the key pools")
    parser.add_argument("--sig_grid", type=int, default=4)
    parser.add_argument("--key", choices=KEY_KINDS, default="grid", help="Raw key (see pace/pi05.py)")
    parser.add_argument("--key_transform", choices=("pca", "center", "none"), default="pca")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--features", type=int, default=0,
                        help="1: also store per-call image tokens (needed by the v2 encoder; robot states are always "
                             "stored)")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--per_task", type=int, default=0,
                        help="Keep at most this many episodes per task, drawn with --sample_seed (0 = all)")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--fit_episodes", type=int, default=0,
                        help="Fit the key transform on this many episodes (seeded random subset; 0 = all). Large "
                             "datasets: the raw keys of all calls would not fit in memory (lerobot/libero_plus: ~30 GB)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy, preprocessor = load_policy(args.policy_path, args.device, n_action_steps=args.n_action_steps)
    stride = args.stride or policy.config.n_action_steps
    dataset = load_dataset(policy, args.dataset_repo_id, args.dataset_root, args.video_backend,
                           parse_camera_map(args.camera_map))

    keep = None
    if args.suite:
        keep = {normalize_instruction(t) for t in suite_instructions(args.suite)}
    if args.tasks_file:
        with open(args.tasks_file) as f:
            listed = {normalize_instruction(t) for t in json.load(f)}
        keep = listed if keep is None else keep & listed

    episodes_meta = dataset.meta.episodes
    starts, ends = list(episodes_meta["dataset_from_index"]), list(episodes_meta["dataset_to_index"])
    all_tasks = episode_tasks(dataset)
    allowed = None
    if args.per_task:  # a fixed random subset of each task's episodes, in episode order
        by_task = collections.defaultdict(list)
        for ep, task in enumerate(all_tasks):
            by_task[normalize_instruction(task)].append(ep)
        rng = np.random.default_rng(args.sample_seed)
        allowed = {int(ep) for eps in by_task.values()
                   for ep in rng.choice(eps, size=min(args.per_task, len(eps)), replace=False)}
    selected = []
    for ep in range(dataset.meta.total_episodes):
        task = all_tasks[ep]
        if keep is not None and normalize_instruction(task) not in keep:
            continue
        if allowed is not None and ep not in allowed:
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

    # The key transform is fitted on the raw keys of --fit_episodes episodes (a seeded random subset, built first);
    # every later episode is transformed as it is built, so the raw keys (64K floats per call) of a large dataset
    # are never all held at once
    order = list(selected)
    n_fit = len(order) if not args.fit_episodes else min(args.fit_episodes, len(order))
    if n_fit < len(order):
        fit_rng = np.random.default_rng(args.sample_seed + 1)
        fit_set = set(fit_rng.choice(len(order), size=n_fit, replace=False).tolist())
        order = [order[i] for i in sorted(fit_set)] + [order[i] for i in range(len(order)) if i not in fit_set]
    loader = DataLoader(dataset, batch_sampler=[frames for _, _, frames in order], num_workers=args.num_workers,
                        collate_fn=default_collate)
    started = time.time()
    transform, fitted = None, False
    collected = []  # (ep, task, frames, keys [T, d] (raw fp16 until the fit, then transformed), actions, sig, features)
    for n, ((ep, task, frames), raw) in enumerate(zip(order, loader), 1):
        batch = preprocessor(raw)
        out = call_keys(policy, batch, args.cameras, args.sig_grid, args.key, features=bool(args.features))
        keys, sigs = out[:2]
        keys = keys.half().cpu()
        raw_dim = keys.shape[1]
        if fitted:
            keys = apply_key_transform(keys, transform)
        features = {"state": batch_state(batch).float().cpu()}
        if args.features:
            features["vis"] = out[2].half().cpu()
        actions = batch[ACTION].float()  # [T, chunk, A] normalized
        assert actions.shape[1] == policy.config.chunk_size, f"action chunk {tuple(actions.shape)}"
        collected.append((ep, task, frames, keys, actions.cpu(), sigs[0].cpu(), features))
        if n == n_fit:
            transform = fit_key_transform(torch.cat([c[3] for c in collected]), args.key_transform, args.pca_dim)
            collected = [(c[0], c[1], c[2], apply_key_transform(c[3], transform), *c[4:]) for c in collected]
            fitted = True
            print(f"[library] key transform ({args.key_transform}) fitted on {n_fit} episodes", flush=True)
        if n % 50 == 0:
            print(f"[library] {n}/{len(order)} episodes ({time.time() - started:.0f}s)", flush=True)

    collected.sort(key=lambda c: c[0])  # episode order, whatever the fit order was
    info = key_info(args.cameras, args.sig_grid, args.key, transform)
    store = ExperienceStore(info=info, key_transform=transform)
    sims = []
    for ep, task, frames, keys, actions, sig, features in collected:
        store.add(make_episode(
            normalize_instruction(task), keys, actions, sig, True, "demo", name=f"demo/ep{ep:05d}",
            meta={"episode_index": ep, "dataset_indices": frames, "task": task, "stride": stride,
                  "dataset": args.dataset_repo_id, "policy_path": args.policy_path},
            **features,
        ))
        if len(keys) > 1:
            sims.append((keys[:-1] * keys[1:]).sum(-1))
    sims = torch.cat(sims)
    print(f"[library] keys: raw dim {raw_dim} -> {store.episodes[0]['keys'].shape[1]} "
          f"({info['transform']}); consecutive-call cosine median {sims.median():.3f} "
          f"(q10 {sims.quantile(0.1):.3f}, q90 {sims.quantile(0.9):.3f})", flush=True)

    lengths = collections.defaultdict(list)
    for episode in store.episodes:
        lengths[episode["task_key"]].append(episode["keys"].shape[0])
    for task, values in sorted(lengths.items()):
        print(f"{len(values):4d} demos, {min(values)}-{max(values)} calls: {task}")
    if args.features:
        ep0 = store.episodes[0]
        print(f"[library] features per call: image tokens {tuple(ep0['vis'].shape[1:])} fp16, state "
              f"{tuple(ep0['state'].shape[1:])}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    store.save(args.out)
    print(f"saved {len(store)} episodes of {len(lengths)} tasks to {args.out} (keys {info})")


if __name__ == "__main__":
    main()
