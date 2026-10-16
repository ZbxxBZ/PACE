"""
Offline probes of a trained adapter on its run's held-out demos: the same split, calls and (noise, time) draws as
pace/train.py's validation, read with different references, so every condition pairs episode by episode.

Conditions:
  real, notext, unaligned, other_task, none   as train.py validates them (on the first 32 episodes these reproduce
                                              the run's train_log at the checkpoint's step: a built-in check)
  self             the demo's own chunk at the same call (a perfect reference: how far reading can go)
  near             real, but near-identical first frames (>= --same_scene_threshold) stay in the pool (same-scene
                   experience, as a store that holds earlier episodes of the same environment offers)
  other_unaligned  the other_task references at random steps (is it the alignment or the task that helps)
  real_top1        only the best real reference
  fixed            one fixed training demo's call for every sample (carries no information about the sample; the
                   adapter never saw such inputs in training, so read it with care)
  novariant        v7 runs (rollouts): real without the other attempts of the sample's own variant (a first round)

    python -m pace.val_probe --run_dir runs/train/v3plus_libero4 --checkpoint adapter_last.pt --val_episodes 64
"""

import argparse
import collections
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from lerobot.utils.constants import ACTION

from .common import PromptMap, load_policy
from .core import load_adapter
from .pi05 import PI05Experience
from .train import (
    Alignments, instruction_sims, load_training_data, make_plan, prepare, sample_frames, scored_loss, split_episodes,
)

BASE = ("real", "notext", "unaligned", "other_task", "none")


def fixed_reference(demos, alignments, query, ref, call):
    """Demo `ref`'s call `call` for every call of demo `query`, with constant scalars (no information on the query)."""
    T = demos.actions[query].shape[0]
    ends, zeros = torch.full((T, 1), call, dtype=torch.long), torch.zeros(T, 1)
    if not alignments.v3:
        return [ref], ends, zeros, zeros, 1
    return [ref], ends, zeros, zeros, 1, {}, alignments._extras(zeros, torch.zeros(1), True, torch.full((T, 1), 0.5))


def near_stats(alignments, refs, episodes, frames, threshold):
    """Per episode: near-identical candidates in the pool, and the fraction of its calls whose best reference is one."""
    counts, top = [], []
    for e, ref, frame in zip(episodes, refs, frames):
        pool = alignments.pool.get(e, [])
        counts.append(sum(float(alignments.sig_sims[e, j]) >= threshold for j in pool))
        if ref is None:
            top.append(0.0)
            continue
        members, order = ref[0], (ref[6]["rank"] if len(ref) > 6 else ref[3])
        best = [members[int(torch.argmin(order[int(f)]))] for f in frame]
        top.append(float(np.mean([float(alignments.sig_sims[e, r]) >= threshold for r in best])))
    return counts, top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="A pace.train output directory (its args.json is reused)")
    parser.add_argument("--checkpoint", default="adapter_last.pt", help="File in --run_dir, or a path")
    parser.add_argument("--val_episodes", type=int, default=64, help="Held-out demos scored (train.py used 32)")
    parser.add_argument("--out", default=None, help="Default: <run_dir>/probe_<checkpoint stem>.json")
    cli = parser.parse_args()

    run_dir = Path(cli.run_dir)
    args = argparse.Namespace(**json.load(open(run_dir / "args.json")))
    args.val_group = getattr(args, "val_group", 0)
    ckpt_path = Path(cli.checkpoint) if Path(cli.checkpoint).exists() else run_dir / cli.checkpoint
    started = time.time()

    policy, preprocessor = load_policy(args.policy_path, "cuda", n_action_steps=args.n_action_steps)
    args.camera_map = getattr(args, "camera_map", [])
    args.rollouts = getattr(args, "rollouts", [])
    store, demos, make_dataset = load_training_data(policy, args)
    dataset = make_dataset()
    prompts = PromptMap(args.prompt_override, args.prompt_map)
    # train.py's split: the first draw of default_rng(seed)
    _, val_indices, train_indices = split_episodes(demos, args, np.random.default_rng(args.seed))
    # build_validation's stream: the episodes first, then per group frames and unaligned steps
    rng = np.random.default_rng(args.seed + 1)
    chosen = [val_indices[i] for i in rng.permutation(len(val_indices))[: cli.val_episodes]]

    task_sims = None
    if args.retrieval == "v3":
        task_sims = instruction_sims(policy, store, sorted(demos.by_task))
    alignments = Alignments(demos, train_indices, args, task_sims, only=chosen)
    args_near = copy.copy(args)
    args_near.same_scene_threshold = 2.0  # keep every candidate, near-identical scenes included
    near_alignments = Alignments(demos, train_indices, args_near, task_sims, only=chosen)

    adapter, ckpt = load_adapter(ckpt_path)
    pace = PI05Experience(policy, adapter, runtime_kwargs={"retrieval": args.retrieval})
    pace.adapter.eval()
    fixed_ref = int(np.random.default_rng(args.seed + 7).choice(train_indices))
    fixed_call = demos.actions[fixed_ref].shape[0] // 2
    print(f"[probe] {ckpt_path} (step {ckpt.get('step')}), {len(chosen)} held-out demos; fixed reference "
          f"{demos.names[fixed_ref]} call {fixed_call}", flush=True)

    rng_x = np.random.default_rng(args.seed + 101)  # extra conditions: the base stream stays train.py's
    group = args.val_group or args.episodes_per_step
    batches, near_counts, near_top = [], [], []
    for start in range(0, len(chosen), group):
        episodes = chosen[start : start + group]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, rng) for e in episodes]
        text = {"text": True} if alignments.v3 else {}
        keep = {"variant": True} if alignments.variants is not None else {}  # v7: as train.py validates
        refs = {c: [alignments.references(e, c, rng, num=args.num_refs, full_pool=True, **text, **keep)
                    for e in episodes]
                for c in ("real", "other_task")}
        if alignments.v3:
            refs["notext"] = [alignments.references(e, "real", rng, num=args.num_refs, full_pool=True, text=False,
                                                    **keep)
                              for e in episodes]
        conditions = {}
        for condition in BASE:
            if (condition == "other_task" and not alignments.has_other_tasks) or \
                    (condition == "notext" and condition not in refs):
                continue
            source = refs[condition] if condition in ("other_task", "notext") else refs["real"]
            conditions[condition] = make_plan(demos, episodes, source, [condition] * len(episodes), frames, rng, args)
        n = len(episodes)
        extra = {
            "self": [alignments.references(e, "self", rng_x, **text) for e in episodes],
            "near": [near_alignments.references(e, "real", rng_x, num=args.num_refs, full_pool=True, **text, **keep)
                     for e in episodes],
            "real_top1": [alignments.references(e, "real", rng_x, num=1, full_pool=True, **text, **keep)
                          for e in episodes],
            "fixed": [fixed_reference(demos, alignments, e, fixed_ref, fixed_call) for e in episodes],
        }
        if keep:  # v7: without the own variant's other attempts (a first round's store)
            extra["novariant"] = [alignments.references(e, "real", rng_x, num=args.num_refs, full_pool=True, **text,
                                                        variant=False) for e in episodes]
        for condition, source in extra.items():
            conditions[condition] = make_plan(demos, episodes, source, [condition] * n, frames, rng_x, args)
        if "other_task" in conditions:
            conditions["other_unaligned"] = make_plan(demos, episodes, refs["other_task"], ["unaligned"] * n, frames,
                                                      rng_x, args)
        counts, top = near_stats(near_alignments, extra["near"], episodes, frames, args.same_scene_threshold)
        near_counts += counts
        near_top += top
        raw = default_collate([dataset[i] for i in conditions["real"]["indices"]])
        batches.append({"conditions": conditions, "raw": raw, "E": n, "chunks": conditions["real"]["chunks"]})

    # train.py's validate: the same (noise, time) draws for every condition
    losses = collections.defaultdict(list)
    relevance = collections.defaultdict(list)  # v6 adapters: mean reference weight g per condition
    refmass = collections.defaultdict(list)  # v6 adapters: the input injection's total weight on the references
    with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed + 2)
        for batch_data in batches:
            batch = prepare(batch_data["raw"], preprocessor, prompts, batch_data.get("chunks"))
            B = batch[ACTION].shape[0]
            shape = (args.repeats, B, policy.config.chunk_size, policy.config.max_action_dim)
            noise = torch.randn(shape, device="cuda")
            time_ = policy.model.sample_time(args.repeats * B, "cuda").view(args.repeats, B)
            for condition, plan in batch_data["conditions"].items():
                pace.adapter.relevance_stats = {}
                loss = scored_loss(policy, pace, batch, None if condition == "none" else plan["exp"], args.repeats,
                                   noise=noise, time=time_)
                losses[condition].append(loss.view(batch_data["E"], args.group_size).mean(dim=1).cpu())
                if "relevance_mean" in pace.adapter.relevance_stats:
                    relevance[condition].append(pace.adapter.relevance_stats["relevance_mean"])
                if "inject_ref_mass" in pace.adapter.relevance_stats:
                    refmass[condition].append(pace.adapter.relevance_stats["inject_ref_mass"])
    per_episode = {c: torch.cat(v).double() for c, v in losses.items()}

    def summary(rows):
        out = {"n_episodes": int(len(rows))}
        for c, v in per_episode.items():
            out[f"val_{c}"] = float(v[rows].mean())
            for base in ("real", "none"):
                if c != base and base in per_episode:
                    d = v[rows] - per_episode[base][rows]
                    out[f"{c}_minus_{base}"] = float(d.mean())
                    out[f"{c}_minus_{base}_se"] = float(d.std() / math.sqrt(len(d))) if len(d) > 1 else float("nan")
        return out

    everything = torch.arange(len(chosen))
    result = {
        "checkpoint": str(ckpt_path), "step": ckpt.get("step"), "val_episodes": len(chosen),
        "fixed_reference": [demos.names[fixed_ref], fixed_call],
        "near_identical": {"threshold": args.same_scene_threshold,
                           "candidates_per_episode": float(np.mean(near_counts)),
                           "episodes_with_any": float(np.mean([c > 0 for c in near_counts])),
                           "calls_best_is_near": float(np.mean(near_top))},
        "relevance": {c: float(np.mean(v)) for c, v in relevance.items()},
        "refmass": {c: float(np.mean(v)) for c, v in refmass.items()},
        "all": summary(everything),
        "first32": summary(everything[:32]),  # = train_log's validation at this step (same draws)
        "seconds": time.time() - started,
    }
    out = Path(cli.out) if cli.out else run_dir / f"probe_{ckpt_path.stem}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    a = result["all"]
    print(f"[probe] near-identical: {result['near_identical']}", flush=True)
    print(f"{'condition':16s} {'loss':>8s} {'- real':>10s} {'se':>8s} {'- none':>9s} {'relevance':>10s} "
          f"{'refmass':>8s}", flush=True)
    for c in per_episode:
        rel, mass = result["relevance"].get(c), result["refmass"].get(c)
        print(f"{c:16s} {a[f'val_{c}']:8.4f} {a.get(f'{c}_minus_real', 0.0):+10.5f} "
              f"{a.get(f'{c}_minus_real_se', 0.0):8.5f} {a.get(f'{c}_minus_none', 0.0):+9.4f} "
              f"{'' if rel is None else f'{rel:10.3f}'} {'' if mass is None else f'{mass:8.3f}'}", flush=True)
    print(f"[probe] first 32 (train_log check): " + json.dumps({k: round(v, 5) for k, v in result['first32'].items()
                                                              if k.startswith('val_')}), flush=True)
    print(f"saved {out} ({result['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
