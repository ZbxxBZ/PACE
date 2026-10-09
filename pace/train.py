"""
Train the PACE adapter on a frozen pi0.5: only the adapter learns, with pi0.5's own flow-matching loss scored on its
inference path (see `pace/pi05.py`).

The demos are the episodes of a library built by `pace/build_library.py` from the training dataset (their per-call
keys, action chunks and dataset frame indices). Every demo is first aligned, with Progressive Memory Alignment (PMA)
call by call as the evaluation runs it, to its pool: the other demos of its task (never itself, nor a near-identical
scene). Every step then:
  - samples --episodes_per_step demos and --group_size calls of each;
  - gives each demo m ~ U{1..--num_refs} references per call: the m best-aligned members of its pool at that call, with
    probability --p_subpool from a random subset of 1..2*num_refs pool members (a store that is still small); with
    probability --p_none none at all (the base policy) and with --p_distractor m demos of another task, aligned the
    same way; with probability --p_near_self the demo itself joins the candidates as a near-identical past episode
    would look (a drifting offset of up to --near_self_offset calls, chunks noised);
  - loads those frames from the dataset, tells the policy --prompt_override / --prompt_map instead of the true
    instruction if given (hidden-goal experiments), caches the prefix without gradients and scores one denoising step
    per --repeats draw of (noise, flow time).
--ref_content `unaligned` trains a control adapter reading a random step of each reference; `self` is a sanity check
where every reference is the demo's own chunk, perfectly aligned (can the adapter copy?).
Every --val_every steps, held-out demos are scored with aligned / unaligned / other-task / no references on identical
(noise, time) draws, so the conditions are compared pairwise.

    python -m pace.train --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
        --library libs/libero_goal_demo.pt --n_action_steps 10 --out_dir runs/goal_main
"""

import argparse
import collections
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from lerobot.utils.constants import ACTION

from .common import PromptMap, load_dataset, load_policy
from .core import SCALAR_NAMES, ExperienceStore, align_sequence, pad_keys, reference_scalars, save_adapter
from .pi05 import PI05Experience, build_adapter, flow_loss, prefix_cache


class Demos:
    """Library demos in host memory: keys, action chunks, signatures and dataset frame indices per call."""

    def __init__(self, store: ExperienceStore, max_episodes: int = 0) -> None:
        episodes = [ep for ep in store.episodes if ep["source"] == "demo"]
        if max_episodes:
            episodes = episodes[:max_episodes]
        self.names = [ep["name"] for ep in episodes]
        self.task = [ep["task_key"] for ep in episodes]
        self.keys = [ep["keys"] for ep in episodes]
        self.actions = [ep["actions"] for ep in episodes]
        self.sig = [ep["sig"] if ep["sig"] is not None else ep["keys"][0] for ep in episodes]
        self.indices = [list(ep["meta"]["dataset_indices"]) for ep in episodes]
        for i, ep in enumerate(episodes):
            assert len(self.indices[i]) == ep["keys"].shape[0], f"{ep['name']}: indices vs keys"
        self.by_task = collections.defaultdict(list)
        for i, task in enumerate(self.task):
            self.by_task[task].append(i)
        print(f"[demos] {len(self)} demos of {len(self.by_task)} tasks", flush=True)

    def __len__(self):
        return len(self.names)


def align(demos, query, members, args):
    """PMA of demo `query` against demos `members`: ends, similarities, mean costs, each [T_query, len(members)] (CPU)."""
    keys, lengths = pad_keys([demos.keys[m] for m in members])
    ends, sims, costs = align_sequence(demos.keys[query].cuda(), keys.cuda(), lengths, args.vmax, args.gamma)
    return ends.cpu(), sims.cpu(), costs.cpu()


class Alignments:
    """PMA of every demo against its pool (training demos of its task, minus itself and near-identical scenes)."""

    def __init__(self, demos, train_indices, args):
        self.demos, self.args = demos, args
        train = set(train_indices)
        sig = torch.stack(demos.sig).float().cuda()
        sig_sims = (sig @ sig.T).cpu()
        self.pool, self.ends, self.sims, self.costs = {}, {}, {}, {}
        dropped, top_costs, spreads, started = [], [], [], time.time()
        for i in range(len(demos)):
            pool = [j for j in demos.by_task[demos.task[i]] if j in train and j != i]
            self.pool[i] = [j for j in pool if sig_sims[i, j] < args.same_scene_threshold]
            dropped.append(len(pool) - len(self.pool[i]))
            if self.pool[i]:
                self.ends[i], self.sims[i], self.costs[i] = align(demos, i, self.pool[i], args)
                top_costs.append(1 - self.sims[i].gather(1, self.costs[i].argmin(dim=1, keepdim=True))[:, 0])
                spreads.append(self.costs[i].median(dim=1).values - self.costs[i].min(dim=1).values)
        self.other = {i: [j for j in train if demos.task[j] != demos.task[i]] for i in range(len(demos))}
        self.has_other_tasks = any(self.other.values())
        assert self.ends, "No demo has a same-task pool; more episodes are needed"
        top_costs, spreads = torch.cat(top_costs).numpy(), torch.cat(spreads).numpy()
        q = lambda x: " ".join(f"q{p}={np.quantile(x, p / 100):.4f}" for p in (10, 50, 90))  # noqa: E731
        print(f"[pma] aligned {len(self.ends)} demos in {time.time() - started:.0f}s; near-identical scenes dropped "
              f"per demo: mean={np.mean(dropped):.2f} max={max(dropped)}", flush=True)
        # Calibration of --gamma: compare it with the per-call cost of the best memory and the cost spread
        print(f"[pma] best-memory per-call cost {q(top_costs)}; median-minus-best mean cost {q(spreads)}; "
              f"gamma={args.gamma}", flush=True)

    def references(self, i, condition, rng, num=None, full_pool=False):
        """
        Candidates for demo i under `condition`: (member demo indices, ends, sims, costs [T, C], m[, noise]). The m
        best-aligned candidates are read at every call.
        """
        args = self.args
        if condition == "none":
            return None
        if condition == "self":  # sanity check (--ref_content self): the demo itself, matched at the same call
            T = self.demos.actions[i].shape[0]
            return [i], torch.arange(T)[:, None], torch.ones(T, 1), torch.zeros(T, 1), 1
        m = num if num is not None else int(rng.integers(1, args.num_refs + 1))
        if condition == "other_task":
            pool = self.other[i]
            if not pool:
                return None
            members = [pool[k] for k in rng.permutation(len(pool))[:m]]
            return (members, *align(self.demos, i, members, args), m)
        pool = self.pool[i]
        if not pool:
            return None
        columns = torch.arange(len(pool))
        if not full_pool and rng.random() < args.p_subpool:
            size = int(rng.integers(1, min(len(pool), 2 * args.num_refs) + 1))
            columns = torch.from_numpy(np.sort(rng.permutation(len(pool))[:size]))
        members = [pool[c] for c in columns.tolist()]
        refs = members, self.ends[i][:, columns], self.sims[i][:, columns], self.costs[i][:, columns], m
        if not full_pool and rng.random() < args.p_near_self:
            ends, sims, costs, sigma = self.near_self(i, rng)
            refs = (members + [i], torch.cat([refs[1], ends], 1), torch.cat([refs[2], sims], 1),
                    torch.cat([refs[3], costs], 1), m, {len(members): sigma})
        return refs

    def near_self(self, i, rng):
        """
        Demo i as a near-identical past episode of itself: matched at f + offset, where the offset starts in
        [-J, J] and drifts by -1/0/+1 per call (J = --near_self_offset; the match stays monotonic), with the similarity
        of the matched keys and the running mean cost as PMA reports them; its chunks get noise sigma ~ U(0, max).
        """
        J, keys = self.args.near_self_offset, self.demos.keys[i].float()
        T = keys.shape[0]
        offset, ends = int(rng.integers(-J, J + 1)), []
        for f in range(T):
            offset = int(np.clip(offset + rng.choice((-1, 0, 1), p=(0.2, 0.6, 0.2)), -J, J))
            ends.append(min(max(f + offset, 0), T - 1))
        ends = torch.tensor(ends)
        sims = (keys * keys[ends]).sum(-1).clamp(-1.0, 1.0)
        costs = torch.cumsum(1 - sims, 0) / torch.arange(1, T + 1)
        return ends[:, None], sims[:, None], costs[:, None], float(rng.uniform(0, self.args.near_self_noise))

    def training_condition(self, rng):
        u = rng.random()
        if u < self.args.p_none:
            return "none"
        if self.has_other_tasks and u < self.args.p_none + self.args.p_distractor:
            return "other_task"
        return self.args.ref_content


def reference_batch(demos, episode, frames, refs, unaligned, rng, max_refs):
    """Adapter inputs for `frames` of `episode` reading `refs` (Alignments.references): [G, max_refs, ...] tensors."""
    G = len(frames)
    actions = torch.zeros(G, max_refs, *demos.actions[episode].shape[1:])
    scalars = torch.zeros(G, max_refs, len(SCALAR_NAMES))
    success = torch.zeros(G, max_refs, dtype=torch.long)
    valid = torch.zeros(G, max_refs, dtype=torch.bool)
    if refs is None:
        return {"actions": actions, "scalars": scalars, "success": success, "valid": valid}
    members, ends, sims, costs, m = refs[:5]
    noise = refs[5] if len(refs) > 5 else {}  # candidate column -> chunk noise sigma (near-self references)
    for g, f in enumerate(frames):
        for slot, c in enumerate(torch.argsort(costs[f])[: min(m, max_refs)].tolist()):
            r = members[c]
            T_r = demos.actions[r].shape[0]
            j, sim = int(ends[f, c]), float(sims[f, c])
            if unaligned:  # control: a random step of the same reference
                j = int(rng.integers(T_r))
                sim = float(demos.keys[r][j].float() @ demos.keys[episode][f].float())
            chunk = demos.actions[r][j]
            if c in noise:
                chunk = chunk + noise[c] * torch.from_numpy(rng.standard_normal(tuple(chunk.shape))).float()
            actions[g, slot] = chunk
            scalars[g, slot] = torch.tensor(reference_scalars(j, sim, float(costs[f, c]), int(f), T_r))
            success[g, slot] = 1
            valid[g, slot] = True
    return {"actions": actions, "scalars": scalars, "success": success, "valid": valid}


def sample_frames(T, group_size, rng):
    if T >= group_size:
        return np.sort(rng.choice(T, group_size, replace=False))
    return np.concatenate([np.arange(T), np.full(group_size - T, T - 1)])  # pad with the last call


def make_plan(demos, episodes, refs, conditions, frames, rng, args):
    """Dataset indices and adapter inputs of the calls `frames[k]` of `episodes[k]` reading `refs[k]`."""
    indices, exp_parts = [], []
    for episode, ref, condition, frame in zip(episodes, refs, conditions, frames):
        exp_parts.append(reference_batch(demos, episode, frame, ref, condition == "unaligned", rng, args.max_refs))
        indices.extend(demos.indices[episode][int(f)] for f in frame)
    exp = {key: torch.cat([p[key] for p in exp_parts]) for key in exp_parts[0]}
    return {"indices": indices, "exp": exp, "conditions": conditions,
            "chunks": torch.cat([demos.actions[e][torch.as_tensor(f)] for e, f in zip(episodes, frames)])}


class PlanSampler:
    """Batch sampler that draws one training plan per step; the main loop pops the plans in the same order."""

    def __init__(self, draw, steps):
        self.draw, self.steps = draw, steps
        self.plans = collections.deque()

    def __iter__(self):
        for _ in range(self.steps):
            plan = self.draw()
            self.plans.append(plan)
            yield plan["indices"]

    def __len__(self):
        return self.steps


def prepare(raw, preprocessor, prompts: PromptMap):
    if prompts:
        raw = dict(raw)
        raw["task"] = [prompts(t) for t in raw["task"]]
    return preprocessor(raw)


def check_chunks(batch, plan, step):
    """The dataset chunks must be the library's (same dataset, normalization and frames)."""
    got = batch[ACTION].float().cpu()
    err = float((got - plan["chunks"]).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"step {step}: dataset action chunks differ from the library's by {err:.4g}; was the "
                           "library built from this dataset with this policy's preprocessor?")


def scored_loss(policy, pace, batch, exp, repeats, noise=None, time=None):
    """Mean over --repeats draws of the per-sample loss [B]; pass noise/time [R, ...] for identical draws."""
    prefix = prefix_cache(policy, batch)
    actions = batch[ACTION].float()
    pace.set_exp(exp)
    try:
        losses = [flow_loss(policy, prefix, actions,
                            noise=None if noise is None else noise[r], time=None if time is None else time[r])
                  for r in range(repeats)]
    finally:
        pace.set_exp(None)
    return torch.stack(losses).mean(dim=0)


def build_validation(demos, alignments, val_indices, dataset, args):
    """
    Fixed held-out batches: identical calls and (noise, time) draws in every condition; aligned and unaligned read
    the same references (the --num_refs best-aligned members of the full pool at every call).
    """
    rng = np.random.default_rng(args.seed + 1)
    chosen = [val_indices[i] for i in rng.permutation(len(val_indices))[: args.val_episodes]]
    batches = []
    for start in range(0, len(chosen), args.episodes_per_step):
        episodes = chosen[start : start + args.episodes_per_step]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, rng) for e in episodes]
        refs = {c: [alignments.references(e, c, rng, num=args.num_refs, full_pool=True) for e in episodes]
                for c in ("real", "other_task")}
        if args.ref_content == "self":
            refs["self"] = [alignments.references(e, "self", rng) for e in episodes]
        conditions = {}
        for condition in ("real", "unaligned", "other_task", "none", "self"):
            if (condition == "other_task" and not alignments.has_other_tasks) or (condition == "self" and "self" not in refs):
                continue
            source = refs[condition] if condition in ("other_task", "self") else refs["real"]
            conditions[condition] = make_plan(demos, episodes, source, [condition] * len(episodes), frames, rng,
                                              args)
        first = next(iter(conditions.values()))
        raw = default_collate([dataset[i] for i in first["indices"]])
        batches.append({"conditions": conditions, "raw": raw, "E": len(episodes)})
    return batches


@torch.no_grad()
def validate(policy, pace, batches, preprocessor, prompts, args):
    pace.adapter.eval()
    losses = collections.defaultdict(list)
    # The same (noise, time) draws in every condition and every validation pass; the training RNG is left untouched
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed + 2)
        for batch_data in batches:
            batch = prepare(batch_data["raw"], preprocessor, prompts)
            B = batch[ACTION].shape[0]
            shape = (args.repeats, B, policy.config.chunk_size, policy.config.max_action_dim)
            noise = torch.randn(shape, device="cuda")
            time_ = policy.model.sample_time(args.repeats * B, "cuda").view(args.repeats, B)
            for condition, plan in batch_data["conditions"].items():
                loss = scored_loss(policy, pace, batch, None if condition == "none" else plan["exp"], args.repeats,
                                   noise=noise, time=time_)
                losses[condition].append(loss.view(batch_data["E"], args.group_size).mean(dim=1).cpu())
    pace.adapter.train()
    per_episode = {c: torch.cat(v).double() for c, v in losses.items()}
    result = {f"val_{c}": float(v.mean()) for c, v in per_episode.items()}
    for base, other in (("real", "none"), ("real", "unaligned"), ("real", "other_task"), ("self", "none")):
        if base in per_episode and other in per_episode:
            diff = per_episode[base] - per_episode[other]
            result[f"{base}_minus_{other}"] = float(diff.mean())
            result[f"{base}_minus_{other}_se"] = float(diff.std() / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--dataset_repo_id", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--library", required=True, help="Demo library of this dataset (pace/build_library.py)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--prompt_override", default=None, help="Tell the policy this instead of every instruction")
    parser.add_argument("--prompt_map", default=None, help="JSON {instruction: prompt} ('*' = default)")
    parser.add_argument("--ref_content", choices=("real", "unaligned", "self"), default="real")
    parser.add_argument("--num_refs", type=int, default=4, help="Maximum references read per call")
    parser.add_argument("--max_refs", type=int, default=8, help="Adapter slots (>= num_refs; deploy may use up to this)")
    parser.add_argument("--vmax", type=int, default=3, help="PMA: largest memory advance per call")
    parser.add_argument("--gamma", type=float, default=0.01,
                        help="PMA: soft prior gamma*|advance-1|, on the scale of the per-call cost 1-cos (see [pma] log)")
    parser.add_argument("--p_none", type=float, default=0.25)
    parser.add_argument("--p_distractor", type=float, default=0.1)
    parser.add_argument("--p_subpool", type=float, default=0.5)
    parser.add_argument("--same_scene_threshold", type=float, default=0.999)
    parser.add_argument("--p_near_self", type=float, default=0.0)
    parser.add_argument("--near_self_offset", type=int, default=2)
    parser.add_argument("--near_self_noise", type=float, default=0.2, help="Largest chunk noise sigma (normalized)")
    parser.add_argument("--inject_every", type=int, default=2, help="Cross-attention after every n-th expert layer")
    parser.add_argument("--input_inject", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--episodes_per_step", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2, help="(noise, time) draws per sample and step")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--val_episodes", type=int, default=32)
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--check_every", type=int, default=200, help="Compare dataset and library chunks (0 = never)")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.num_refs <= args.max_refs

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    policy, preprocessor = load_policy(args.policy_path, "cuda", n_action_steps=args.n_action_steps)
    dataset = load_dataset(policy, args.dataset_repo_id, args.dataset_root, args.video_backend)
    prompts = PromptMap(args.prompt_override, args.prompt_map)
    store = ExperienceStore.load(args.library)
    demos = Demos(store, args.max_episodes)
    permutation = rng.permutation(len(demos)).tolist()
    val_indices = permutation[: max(1, int(len(permutation) * args.val_frac))]
    held_out = set(val_indices)
    train_indices = [i for i in permutation if i not in held_out]
    alignments = Alignments(demos, train_indices, args)
    print(f"episodes: train={len(train_indices)} val={len(val_indices)}", flush=True)

    adapter = build_adapter(policy, inject_every=args.inject_every, max_refs=args.max_refs,
                            input_inject=bool(args.input_inject))
    pace = PI05Experience(policy, adapter, cameras=store.info.get("cameras", (0, 1)),
                          sig_grid=store.info.get("sig_grid", 4))
    adapter.train()
    params = [p for p in adapter.parameters() if p.requires_grad]
    print(f"adapter parameters: {sum(p.numel() for p in params) / 1e6:.1f}M, layers after expert blocks "
          f"{adapter.inject_layers}, config {adapter.config}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    val_batches = build_validation(demos, alignments, val_indices, dataset, args)
    log = open(out_dir / "train_log.jsonl", "a")

    def record(row):
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    def draw():
        episodes = [train_indices[k] for k in rng.integers(len(train_indices), size=args.episodes_per_step)]
        conditions = [alignments.training_condition(rng) for _ in episodes]
        refs = [alignments.references(e, "real" if c == "unaligned" else c, rng) for e, c in zip(episodes, conditions)]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, rng) for e in episodes]
        return make_plan(demos, episodes, refs, conditions, frames, rng, args)

    sampler = PlanSampler(draw, args.steps)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=default_collate,
                        persistent_workers=False)

    val = validate(policy, pace, val_batches, preprocessor, prompts, args)
    record({"step": 0, **val})  # zero-initialized adapter: every condition equals `none` here
    running, started = collections.defaultdict(float), time.time()
    for step, raw in enumerate(loader, 1):
        plan = sampler.plans.popleft()
        batch = prepare(raw, preprocessor, prompts)
        if args.check_every and (step == 1 or step % args.check_every == 0):
            check_chunks(batch, plan, step)
        loss = scored_loss(policy, pace, batch, plan["exp"], args.repeats).mean()
        optimizer.zero_grad(set_to_none=True)
        if loss.requires_grad:  # a batch without any reference has no path to the adapter
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            optimizer.step()
        else:
            grad_norm = 0.0
            running["n_skipped"] += 1
        scheduler.step()

        running["loss"] += float(loss)
        for condition in plan["conditions"]:
            running[f"n_{condition}"] += 1
        if step % args.log_every == 0:
            gates = adapter.gates()
            record({
                "step": step, "loss": running.pop("loss") / args.log_every, "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm, "steps_per_s": step / (time.time() - started),
                "gate_mean": float(np.mean(list(gates.values()))), "gate_max": float(np.max(np.abs(list(gates.values())))),
                **running,
            })
            running.clear()
        if step % args.val_every == 0 or step == args.steps:
            val = validate(policy, pace, val_batches, preprocessor, prompts, args)
            record({"step": step, **val, **adapter.gates()})
        if step % args.save_every == 0 or step == args.steps:
            extra = {"step": step, "args": vars(args), "val": val, "policy_path": args.policy_path,
                     "key_info": store.info, "vmax": args.vmax, "gamma": args.gamma}
            save_adapter(adapter, out_dir / f"adapter_step{step:06d}.pt", **extra)
            save_adapter(adapter, out_dir / "adapter_last.pt", **extra)
    log.close()


if __name__ == "__main__":
    main()
