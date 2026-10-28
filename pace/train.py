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
--adhere adds a contrastive term to the BC loss on the samples reading real references, against a negative
reference set of the same sample (with probability --p_neg_other the best-aligned demos of another task, else the same
references read at random steps), on identical (noise, time) draws:
  pace    rank:  relu(log L_real - log sg(L_neg) + --adhere_margin)  x --adhere_rank   (real must beat the negative by
                 a relative margin; only L_real is pushed, so the adapter cannot win by spoiling the negative)
          harm:  relu(log L_neg - log L_base)                         x --adhere_harm   (a useless reference must not
                 make things worse than the frozen base, L_base = the policy without references)
  margin  RA-VLA's contextual adherence (arXiv 2608.25585): relu(m - (L_neg - L_real)) x --adhere_lambda, m =
          --adhere_abs_margin, gradients through both branches
  none    plain BC
--inject_layers / --position / --time_cond choose where and how the adapter enters the action expert (see
`pace/pi05.py`). --encoder chooses how references become tokens (`core.ExperienceAdapter`): v1 (action chunks and
scalars) or v2 (also the references' and the current call's image tokens and robot states, through a small
transformer: --enc_dim / --enc_layers / --enc_heads / --act_segments); v2 needs a library built with --features 1,
whose stored current-call features are used for the training samples (no extra image pass).
--retrieval chooses how references are found (it fixes the adapter's per-reference scalars): classic (the demo's own
task, image keys; v1 and v2 were trained so) or v3 (`core.ExperienceRuntime`): the candidates are the demos of the
tasks closest to the demo's instruction (`pi05.InstructionText`) and first-frame scene, the alignment cost mixes in
robot states (--state_weight / --state_sigma), the ranking adds --text_weight x (1 - instruction similarity), the
instruction is withheld from the retrieval of a sample with probability --p_text_drop, distractors and adherence
negatives are the best-aligned candidates of other tasks (--hard_distractor), and the adapter also reads the task
similarity, whether an instruction was used and the state cost; validation adds `notext` (real references retrieved
without the instruction). v3 needs a library with robot states (build_library stores them).
--generic mlp (v4) adds a reference-independent branch acting on every sample (`core.GenericResidual`), so the plain
fine-tune any adapter can learn on the training data goes there instead of through the references; `none` validation
is then the host plus that branch, and real - none is what the references add. --train_parts runs it in two stages:
`generic` (every sample without references, only the generic branch learns), then `reference` (--init_from the
stage-1 adapter, generic branch frozen, only the reference branch learns); `all` trains both jointly.
--relevance gate (v6, with the generic branch, both stages) lets the reference branch decline references (see
`core.ExperienceAdapter`); it learns when from a mix of samples: good references (retrieved; --p_near_self:
near-identical experience), and bad ones it should decline: --p_misaligned (read at random steps), --p_distractor
(other tasks), --p_sparse (a sparse, loosely related store: a few random demos of the own task's other scenes or of
other tasks, as a closed-loop store offers early on). Validation then also reports `sparse`, the mean reference
weight g per condition (`relevance_<condition>`) and the input injection's total weight on the references
(`refmass_<condition>`); the training log averages both over the logged steps' own (not the negatives') references.
In stage 2 --p_none only costs time: a sample without references has no path to the reference branch.
--rollouts (v7) trains on the policy's own rollouts instead of demos (recorded by pace.eval_suite --record_frames, see
`pace/rollouts.py`): the targets are the successful attempts' executed action chunks, read on the recorded frames;
whole variants are held out for validation; the references are the training variants' attempts (failed ones too, with
their outcome, with --ref_failures 1), and a sample's other attempts of its own variant stay candidates with
probability --p_same_variant (a closed-loop store holds them from the second round on; validation: `real` with them,
`novariant` without). No generic branch is needed: the host fits its own rollouts, so declining costs nothing.
--val_stratify 1 holds out variants in proportion per category, difficulty and outcome (all / some / no attempts
succeeded); --mixed_weight w draws the successes of mixed variants w times as often (where a successful reference can
change the outcome); --max_same_variant k keeps 1..k of the own variant's attempts (a 3-round evaluation stores 1 or
2). Validation also reports val10_* (the first n_action_steps steps, what the closed loop executes; the best adapter
is picked by val10_real) and val10_mixed_* (held-out mixed variants).
v8 targets what v7 (B: --loss_steps 10) learned instead of experience - every reference, other tasks' too, lowered the
loss about as much as the real ones: --ref_isolation 1 keeps the current call out of the reference tokens (the channel
for a reference-independent correction), --anchor_bad w makes samples with wrong references (other task, misaligned)
match the host in either direction, --adhere_symmetric 1 does the same for the adherence negatives.
v9 adds corrective samples (--p_corrective q): v8 learned to follow the aligned references, which at a failing attempt's
observation reinforces the failure's own way (the references aligned to its state show what it does anyway). Where a
mixed variant's success S and failure F part (PMA-aligned states, `corrective_items`: the first call whose aligned
state distance exceeds --corr_thr twice in a row), the --corr_window calls before it give samples of F's observation
with S's aligned action as the target and the usual references (successes only, as at evaluation; --anchor_bad and the
adherence negatives apply as to any sample), q of each step's episodes. Validation adds the corrective test on the
held-out mixed variants: at F's observation, the loss of S's and of F's own action per condition (val_corr_*); a
reference that tells which way to go lowers the first more than the second (val_corr_<c>_gain > 0), and
adapter_best_corr.pt keeps the largest real gain; val_succ_* is the same test at the successes' observations (their
own way must stay preferred: gain not below 0). --corr_relative 1 trains a corrective sample on its loss relative to
the host's (times --corr_scale): absolute, the few corrective calls (host loss ~25x a plain call's) took ~90% of the
gradient and the adapter learned a reference-independent shift that already hurt the plain held-out loss.
--corr_label branch takes the corrective targets from counterfactual branching instead (`branch_items`, --branches):
F's env replayed exactly up to a call j just before it parts, then the base policy continued K times; the executed
actions of each branch that succeeded are a causal label for F's observation at j (aligned success actions were only
a proxy: ahead of F's progress in ~2/3 of the items). val_corr_* then tests the held-out failures' branch labels;
val_succ_* keeps the same-phase items' success side.
v9 v2 (--recovery, --p_recovery q): the branch-call labels carry no information (a failure's own action there is as
close to the successful branches' as they are to each other; the branches part a median 8 calls later), so the
corrective signal becomes the successful branches' whole continuations, rendered into the rollout format
(render_branches.py): recovery samples (`recovery_items`) are a successful branch's calls after its branch point, on
its own observations, with its executed chunks as targets, drawn around where the branches part (--rec_window) and
equally per branch point; q of each step's episodes, references and conditions as for any sample. --rec_contrast w
adds a bounded preference for the successful branch's action over the failed branch's at the same call. Rendered
branches are never plain targets, references or validation episodes. Validation adds the recovery test (val_rec_*,
held-out variants' branch points; adapter_best_rec.pt keeps the largest real gain).
--ref_content `unaligned` trains a control adapter reading a random step of each reference; `self`
is a sanity check where every reference is the demo's own chunk, perfectly aligned (can the adapter copy?).
Every --val_every steps, held-out demos are scored with aligned / unaligned / other-task / no references on identical
(noise, time) draws, so the conditions are compared pairwise.

    python -m pace.train --policy_path lerobot/pi05_libero_finetuned --dataset_repo_id lerobot/libero \
        --library libs/libero_goal_demo.pt --n_action_steps 10 --out_dir runs/goal_main

Several GPUs: launch with `torchrun --nproc_per_node N -m pace.train ...` (plain data parallelism of the adapter: every
process holds the frozen policy on its GPU and draws its own batch with its own seed, the adapter gradients are averaged
before the optimizer step, so a step uses N x the batch of one GPU; process 0 validates, logs and saves).
"""

import argparse
import collections
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, default_collate

from lerobot.utils.constants import ACTION, OBS_STATE

from .common import PromptMap, load_dataset, load_policy, parse_camera_map
from .core import (
    SCALAR_NAMES, V3_SCALAR_NAMES, ExperienceStore, align_sequence, choose_tasks, has_features, has_states, pad_keys,
    reference_scalars, save_adapter, state_cost,
)
from .pi05 import InstructionText, PI05Experience, batch_state, build_adapter, flow_loss, prefix_cache
from .rollouts import RolloutFrames, frame_files, rollout_store_files


class Demos:
    """
    Library demos in host memory: keys, action chunks, signatures and dataset frame indices per call, and (libraries
    built with --features 1) image tokens and robot states per call. v7 (`frames`, a `rollouts.RolloutFrames`): the
    store's rollouts instead, with their outcomes and variants (the eval_suite task id), the frame indices from
    `frames` (rollouts it did not record are left out) and the executed action chunks as the targets.
    """

    def __init__(self, store: ExperienceStore, max_episodes: int = 0, frames=None) -> None:
        if frames is None:
            episodes = [ep for ep in store.episodes if ep["source"] == "demo"]
        else:
            latest = {ep["name"]: ep for ep in store.episodes if ep["source"] == "rollout"}  # a resumed batch: the last
            episodes = [ep for ep in latest.values() if ep["name"] in frames.index]
            missing = len(latest) - len(episodes)
            if missing:
                print(f"[demos] {missing} rollouts have no recorded frames and are left out", flush=True)
        if max_episodes:
            episodes = episodes[:max_episodes]
        self.rollouts = frames is not None
        self.names = [ep["name"] for ep in episodes]
        self.task = [ep["task_key"] for ep in episodes]
        self.keys = [ep["keys"] for ep in episodes]
        self.actions = [ep["actions"] for ep in episodes]
        self.sig = [ep["sig"] if ep["sig"] is not None else ep["keys"][0] for ep in episodes]
        self.success = [bool(ep["success"]) for ep in episodes]
        self.variant = self.stratum = None
        # v9 v2 (--recovery): rendered counterfactual branches (render_branches.py), (F's name, branch call) or None;
        # never targets, references or validation episodes of their own (split_episodes), only recovery samples
        self.branch = [None] * len(episodes)
        if self.rollouts:
            self.indices = [frames.index[ep["name"]] for ep in episodes]
            self.variant = [int(ep["meta"]["task_id"]) for ep in episodes]
            # the variant's LIBERO-plus category and difficulty (eval_suite's row fields), for a stratified split
            self.stratum = [(ep["meta"].get("category"), ep["meta"].get("difficulty")) for ep in episodes]
            self.branch = [(ep["meta"]["branch_of"], int(ep["meta"]["branch_call"])) if ep["meta"].get("branch_of")
                           else None for ep in episodes]
        else:
            self.indices = [list(ep["meta"]["dataset_indices"]) for ep in episodes]
        for i, ep in enumerate(episodes):
            assert len(self.indices[i]) == ep["keys"].shape[0], f"{ep['name']}: indices vs keys"
        self.features = all(has_features(ep) for ep in episodes)
        self.has_state = all(has_states(ep) for ep in episodes)
        self.vis = [ep["vis"] for ep in episodes] if self.features else None
        self.state = [ep["state"] for ep in episodes] if self.has_state else None
        self.by_task = collections.defaultdict(list)
        for i, task in enumerate(self.task):
            self.by_task[task].append(i)
        if self.rollouts:
            own = [i for i in range(len(self)) if self.branch[i] is None]
            print(f"[demos] rollouts: {sum(self.success[i] for i in own)} successes of {len(own)}, "
                  f"{len(set(self.variant))} variants, {sum(len(self.indices[i]) for i in own)} calls", flush=True)
            if len(own) < len(self):
                rendered = [i for i in range(len(self)) if self.branch[i] is not None]
                print(f"[demos] rendered branches: {len(rendered)} ({sum(self.success[i] for i in rendered)} "
                      f"successful) of {len({self.branch[i] for i in rendered})} branch points", flush=True)
        print(f"[demos] {len(self)} {'rollouts' if self.rollouts else 'demos'} of {len(self.by_task)} tasks"
              + (f"; features: image tokens {tuple(self.vis[0].shape[1:])}, state {self.state[0].shape[1]}"
                 if self.features else "; no image tokens (v1 encoder only)")
              + ("" if self.has_state else "; no robot states (classic retrieval only)"), flush=True)

    def __len__(self):
        return len(self.names)

    def feature_shapes(self):
        """((V, Dv), Ds), or None without features."""
        return (tuple(self.vis[0].shape[1:]), int(self.state[0].shape[1])) if self.features else None


def align(demos, query, members, args):
    """
    PMA of demo `query` against demos `members`: ends, similarities, mean costs, each [T_query, len(members)] (CPU);
    with the v3 retrieval also the state costs at the aligned steps (robot states join the local cost).
    """
    keys, lengths = pad_keys([demos.keys[m] for m in members])
    if args.retrieval == "v3":
        states = pad_keys([demos.state[m] for m in members])[0]
        out = align_sequence(demos.keys[query].cuda(), keys.cuda(), lengths, args.vmax, args.gamma, args.align,
                             query_states=demos.state[query].cuda(), states=states.cuda(),
                             state_weight=args.state_weight, state_sigma=args.state_sigma)
        return tuple(x.cpu() for x in out)
    ends, sims, costs = align_sequence(demos.keys[query].cuda(), keys.cuda(), lengths, args.vmax, args.gamma,
                                       args.align)
    return ends.cpu(), sims.cpu(), costs.cpu()


def v3_pool(i, train, task_ids, chosen, sims, threshold, per_task=0):
    """
    v3 candidates of demo i: the training demos (`train`, ascending) of the `chosen` tasks except i itself and
    near-identical first frames (similarity >= threshold); with `per_task`, at most that many per task, those with the
    most similar first frames (ties in demo order). Returns (ascending demo indices, number of near-identical dropped).
    Vectorized: a large library (libero_plus: 14347 demos) made the per-element loop take most of an hour.
    """
    cand = train[torch.isin(task_ids[train], torch.tensor(sorted(chosen))) & (train != i)]
    near = sims[cand] >= threshold
    cand = cand[~near]
    if per_task and len(cand):
        order = torch.sort(sims[cand].float(), descending=True, stable=True).indices
        ranked, ranked_tasks = cand[order], task_ids[cand[order]]
        keep = torch.zeros(len(ranked), dtype=torch.bool)
        for t in ranked_tasks.unique():
            keep[(ranked_tasks == t).nonzero()[:per_task, 0]] = True
        cand = ranked[keep].sort().values
    return cand.tolist(), int(near.sum())


class Alignments:
    """
    PMA of every demo against its pool (training demos of its task, minus itself and near-identical scenes). With the
    v3 retrieval (`task_sims` [K, K] = instruction similarities of the sorted tasks) the pool holds the training demos
    of the tasks `choose_tasks` picks for the demo with and without its instruction, aligned with robot states.
    """

    def __init__(self, demos, train_indices, args, task_sims=None, only=None):
        # `only`: align just these demos (pace.val_probe: the validation episodes); default every demo
        self.demos, self.args = demos, args
        self.v3 = args.retrieval == "v3"
        # v7 rollouts: a query's other attempts of its own variant are candidates even when held out (as the store of
        # a later round holds them); failed attempts only with --ref_failures 1
        self.variants = torch.tensor(demos.variant) if demos.variant is not None else None
        # v9 v2: rendered branches are never references (the store at evaluation holds recorded attempts only)
        self.referable = torch.tensor([(s or bool(getattr(args, "ref_failures", 0))) and b is None
                                       for s, b in zip(demos.success, demos.branch)])
        assert self.variants is None or self.v3, "rollouts (v7) need the v3 retrieval"
        train = set(train_indices)
        sig = torch.stack(demos.sig).float().cuda()
        sig_sims = (sig @ sig.T).cpu()
        self.sig_sims = sig_sims
        queries = range(len(demos)) if only is None else sorted(only)
        if self.v3:
            self._init_v3(train_indices, sig_sims, task_sims, queries)
            return
        self.pool, self.ends, self.sims, self.costs = {}, {}, {}, {}
        dropped, top_costs, spreads, started = [], [], [], time.time()
        for i in queries:
            pool = [j for j in demos.by_task[demos.task[i]] if j in train and j != i]
            self.pool[i] = [j for j in pool if sig_sims[i, j] < args.same_scene_threshold]
            dropped.append(len(pool) - len(self.pool[i]))
            if self.pool[i]:
                self.ends[i], self.sims[i], self.costs[i] = align(demos, i, self.pool[i], args)
                top_costs.append(1 - self.sims[i].gather(1, self.costs[i].argmin(dim=1, keepdim=True))[:, 0])
                spreads.append(self.costs[i].median(dim=1).values - self.costs[i].min(dim=1).values)
        others = {t: [j for j in train if demos.task[j] != t] for t in set(demos.task)}  # shared per task
        self.other = {i: others[demos.task[i]] for i in range(len(demos))}
        self.has_other_tasks = any(self.other.values())
        assert self.ends, "No demo has a same-task pool; more episodes are needed"
        top_costs, spreads = torch.cat(top_costs).numpy(), torch.cat(spreads).numpy()
        q = lambda x: " ".join(f"q{p}={np.quantile(x, p / 100):.4f}" for p in (10, 50, 90))  # noqa: E731
        print(f"[pma] aligned {len(self.ends)} demos in {time.time() - started:.0f}s; near-identical scenes dropped "
              f"per demo: mean={np.mean(dropped):.2f} max={max(dropped)}", flush=True)
        # Calibration of --gamma: compare it with the per-call cost of the best memory and the cost spread
        print(f"[pma] best-memory per-call cost {q(top_costs)}; median-minus-best mean cost {q(spreads)}; "
              f"gamma={args.gamma}", flush=True)

    def _init_v3(self, train_indices, sig_sims, task_sims, queries):
        demos, args = self.demos, self.args
        tasks = sorted(demos.by_task)
        index = {t: k for k, t in enumerate(tasks)}
        assert task_sims is not None and task_sims.shape == (len(tasks), len(tasks)), "v3 needs the task similarities"
        self.task_sims = task_sims.float().cpu()
        self.task_ids = torch.tensor([index[t] for t in demos.task])
        train = torch.tensor(sorted(train_indices))
        self.pool, self.ends, self.sims, self.costs, self.scost = {}, {}, {}, {}, {}
        self.text_cols, self.notext_cols, self.member_text, self.same_task = {}, {}, {}, {}
        self.same_variant = {}  # v7: pool members that are attempts of the query's own variant
        dropped, sizes, n_tasks, started = [], [], [], time.time()
        for i in queries:
            ti = int(self.task_ids[i])
            universe = train
            if self.variants is not None:  # + the other referable attempts of its own variant (held out or not)
                own = torch.nonzero((self.variants == self.variants[i]) & self.referable)[:, 0]
                universe = torch.unique(torch.cat([train, own]))
            s = sig_sims[i, universe].clone().float()
            s[universe == i] = float("-inf")  # its own first frame says nothing about the other demos
            scene = torch.full((len(tasks),), float("-inf")).scatter_reduce(0, self.task_ids[universe], s, "amax")
            with_text = choose_tasks(self.task_sims[ti], scene, args.text_tasks, args.scene_tasks)
            without = choose_tasks(None, scene, args.text_tasks, args.scene_tasks)
            chosen = set(with_text) | set(without)
            self.pool[i], n_dropped = v3_pool(i, universe, self.task_ids, chosen, sig_sims[i],
                                              args.same_scene_threshold, args.pool_per_task)
            dropped.append(n_dropped)
            if not self.pool[i]:
                continue
            self.ends[i], self.sims[i], self.costs[i], self.scost[i] = align(demos, i, self.pool[i], args)
            member_tasks = self.task_ids[torch.tensor(self.pool[i])]
            self.text_cols[i] = torch.isin(member_tasks, torch.tensor(with_text))
            self.notext_cols[i] = torch.isin(member_tasks, torch.tensor(without))
            self.member_text[i] = self.task_sims[ti, member_tasks]
            self.same_task[i] = member_tasks == ti
            if self.variants is not None:
                self.same_variant[i] = self.variants[torch.tensor(self.pool[i])] == self.variants[i]
            sizes.append(len(self.pool[i]))
            n_tasks.append(len(chosen))
        others = {t: [j for j in train_indices if demos.task[j] != t] for t in set(demos.task)}  # shared per task
        self.other = {i: others[demos.task[i]] for i in range(len(demos))}
        self.has_other_tasks = any(self.other.values())
        # v6 sparse-store samples: the training demos of each task, in demo order
        self.train_by_task = collections.defaultdict(list)
        for j in sorted(train_indices):
            self.train_by_task[demos.task[j]].append(j)
        assert self.ends, "No demo has a pool; more episodes are needed"
        # How often the top reference comes from the demo's own task, with and without its instruction
        hit_text, hit_without, best_image, best_state = [], [], [], []
        for i in self.ends:
            rank = (self.costs[i] + args.text_weight * (1.0 - self.member_text[i])).masked_fill(
                ~self.text_cols[i], float("inf")).argmin(dim=1)
            hit_text.append(self.same_task[i][rank].float())
            alone = self.costs[i].masked_fill(~self.notext_cols[i], float("inf")).argmin(dim=1)
            hit_without.append(self.same_task[i][alone].float())
            best_image.append(1 - self.sims[i].gather(1, rank[:, None])[:, 0])
            best_state.append(self.scost[i].gather(1, rank[:, None])[:, 0])
        q = lambda x: " ".join(f"q{p}={np.quantile(x, p / 100):.3f}" for p in (10, 50, 90))  # noqa: E731
        print(f"[v3] aligned {len(self.ends)} demos in {time.time() - started:.0f}s: pool {np.mean(sizes):.0f} demos of "
              f"{np.mean(n_tasks):.1f} tasks on average (near-identical scenes dropped {np.mean(dropped):.2f}); top "
              f"reference from the demo's own task: {torch.cat(hit_text).mean():.3f} with its instruction, "
              f"{torch.cat(hit_without).mean():.3f} without", flush=True)
        print(f"[v3] top reference per call: image cost {q(torch.cat(best_image).numpy())}; state cost "
              f"{q(torch.cat(best_state).numpy())} (state_weight={args.state_weight}, state_sigma={args.state_sigma}, "
              f"text_weight={args.text_weight})", flush=True)
        if self.variants is not None:
            counts = [int(v.sum()) for v in self.same_variant.values()]
            fails = [int((~torch.tensor([demos.success[j] for j in self.pool[i]]))[self.same_variant[i]].sum())
                     for i in self.same_variant]
            print(f"[v7] own-variant attempts per pool: mean {np.mean(counts):.2f}, pools with any "
                  f"{np.mean([c > 0 for c in counts]):.3f}; of them failed: mean {np.mean(fails):.2f} "
                  f"(ref_failures={getattr(args, 'ref_failures', 0)}, p_same_variant={args.p_same_variant})", flush=True)

    def references(self, i, condition, rng, num=None, full_pool=False, text=None, variant=None):
        """
        Candidates for demo i under `condition`: (member demo indices, ends, sims, costs [T, C], m[, noise[, extras]]).
        The m best-aligned candidates are read at every call. v3 (`_references_v3`): `text` = whether the instruction
        is used (None: withheld with probability --p_text_drop); v7 rollouts: `variant` = whether the other attempts
        of its own variant stay candidates (None: with probability --p_same_variant; a closed-loop store holds them
        from the second round on).
        """
        if self.v3:
            return self._references_v3(i, condition, rng, num, full_pool, text, variant)
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

    def _extras(self, rank, task_sim, text, scost):
        """v3 inputs of reference_batch: the ranking [T, C] and what feeds the V3_SCALAR_NAMES extras."""
        return {"rank": rank, "task_sim": task_sim if text else torch.zeros_like(task_sim), "text_known": float(text),
                "state_cost": scost, "sigma": self.args.state_sigma}

    def _references_v3(self, i, condition, rng, num, full_pool, text, variant=None):
        """
        v3 candidates, as the evaluation runtime ranks them: real = the pool's members of the tasks chosen with the
        instruction (`text`) or without it, ranked by mean alignment cost + text_weight * (1 - task similarity);
        other_task (distractors, adherence negatives) = with --hard_distractor the best-aligned members of OTHER tasks
        among those candidates (the mistakes the retrieval itself would make), else random other-task demos.
        """
        args, demos = self.args, self.demos
        if condition == "none":
            return None
        if text is None:
            text = bool(rng.random() >= args.p_text_drop)
        if condition == "self":
            T = demos.actions[i].shape[0]
            return ([i], torch.arange(T)[:, None], torch.ones(T, 1), torch.zeros(T, 1), 1, {},
                    self._extras(torch.zeros(T, 1), torch.ones(1), text, torch.zeros(T, 1)))
        m = num if num is not None else int(rng.integers(1, args.num_refs + 1))
        if condition == "sparse":
            return self._sparse_references(i, rng, m, text)
        pool = self.pool[i]
        tables = lambda cols: tuple(x[:, cols] for x in (self.ends[i], self.sims[i], self.costs[i], self.scost[i]))  # noqa: E731
        if condition == "other_task":
            columns = torch.zeros(0, dtype=torch.long)
            if pool and args.hard_distractor:
                columns = torch.nonzero(~self.same_task[i] & (self.text_cols[i] if text else self.notext_cols[i]))[:, 0]
            if len(columns):
                members = [pool[c] for c in columns.tolist()]
                ends, sims, costs, scost = tables(columns)
                task_sim = self.member_text[i][columns]
            else:
                others = self.other[i]
                if not others:
                    return None
                members = [others[k] for k in rng.permutation(len(others))[:m]]
                ends, sims, costs, scost = align(demos, i, members, args)
                task_sim = self.task_sims[int(self.task_ids[i]), self.task_ids[torch.tensor(members)]]
            return members, ends, sims, costs, m, {}, self._extras(costs, task_sim, text, scost)
        if not pool:
            return None
        columns = torch.nonzero(self.text_cols[i] if text else self.notext_cols[i])[:, 0]
        if self.variants is not None:
            if variant is None:
                variant = bool(rng.random() < args.p_same_variant)
            if not variant:  # as in a first round: no attempt of this variant is stored yet
                columns = columns[~self.same_variant[i][columns]]
            elif getattr(args, "max_same_variant", 0) > 0:
                # as in rounds 2 / 3 of a closed-loop run: 1..max_same_variant earlier attempts of the variant stored
                own = self.same_variant[i][columns]
                k = int(rng.integers(1, args.max_same_variant + 1))
                if int(own.sum()) > k:
                    same = columns[own]
                    keep = same[torch.from_numpy(np.sort(rng.permutation(len(same))[:k]))]
                    columns = torch.sort(torch.cat([columns[~own], keep])).values
        if not len(columns):
            return None
        if not full_pool and rng.random() < args.p_subpool:
            size = int(rng.integers(1, min(len(columns), 2 * args.num_refs) + 1))
            columns = columns[torch.from_numpy(np.sort(rng.permutation(len(columns))[:size]))]
        members = [pool[c] for c in columns.tolist()]
        ends, sims, costs, scost = tables(columns)
        task_sim = self.member_text[i][columns]
        rank = costs + args.text_weight * (1.0 - task_sim)[None] if text else costs
        noise = {}
        if not full_pool and rng.random() < args.p_near_self:
            e, s, _, sigma = self.near_self(i, rng)
            sc = state_cost(demos.state[i], demos.state[i][e[:, 0]], args.state_sigma)[:, None]
            local = (1 - args.state_weight) * (1 - s) + args.state_weight * sc
            c = torch.cumsum(local, 0) / torch.arange(1, len(local) + 1)[:, None]
            noise = {len(members): sigma}
            members = members + [i]
            ends, sims, costs, scost = (torch.cat([a, b], 1) for a, b in ((ends, e), (sims, s), (costs, c), (scost, sc)))
            task_sim, rank = torch.cat([task_sim, torch.ones(1)]), torch.cat([rank, c], 1)
        return members, ends, sims, costs, m, noise, self._extras(rank, task_sim, text, scost)

    def _sparse_references(self, i, rng, m, text):
        """
        v6 sparse-store sample: what a store holding only a few, loosely related episodes offers (closed loop, early
        on): 1..--sparse_max demos drawn at random - of demo i's own task with probability --sparse_same_task (its
        near-identical scenes excluded, but NOT the most similar first frames as the pool takes them), else of any
        other task - aligned and ranked as the runtime does. The target is unchanged: the relevance gate has to judge.
        """
        args, demos = self.args, self.demos
        own = [j for j in self.train_by_task[demos.task[i]]
               if j != i and float(self.sig_sims[i, j]) < args.same_scene_threshold
               and (self.variants is None or self.variants[j] != self.variants[i])]  # v7: other variants only
        others = self.other[i]
        members = []
        for _ in range(int(rng.integers(1, args.sparse_max + 1))):
            source = own if (own and (not others or rng.random() < args.sparse_same_task)) else others
            if source:
                members.append(source[int(rng.integers(len(source)))])
        members = list(dict.fromkeys(members))  # drop repeats, keep the draw order
        if not members:
            return None
        ends, sims, costs, scost = align(demos, i, members, args)
        task_sim = self.task_sims[int(self.task_ids[i]), self.task_ids[torch.tensor(members)]]
        rank = costs + args.text_weight * (1.0 - task_sim)[None] if text else costs
        return members, ends, sims, costs, m, {}, self._extras(rank, task_sim, text, scost)

    def training_condition(self, rng):
        u = rng.random()
        if u < self.args.p_none:
            return "none"
        if self.has_other_tasks and u < self.args.p_none + self.args.p_distractor:
            return "other_task"
        # v6: references read at random steps, i.e. unlike the current call (a relevance gate should decline them)
        p_misaligned = getattr(self.args, "p_misaligned", 0.0)
        if u < self.args.p_none + self.args.p_distractor + p_misaligned:
            return "unaligned"
        # v6: references from a sparse, loosely related store (`_sparse_references`)
        if self.v3 and u < self.args.p_none + self.args.p_distractor + p_misaligned + getattr(self.args, "p_sparse", 0.0):
            return "sparse"
        return self.args.ref_content


def reference_batch(demos, episode, frames, refs, unaligned, rng, max_refs, features=False,
                    n_scalars=len(SCALAR_NAMES)):
    """
    Adapter inputs for `frames` of `episode` reading `refs` (Alignments.references): [G, max_refs, ...] tensors; with
    `features` also the references' image tokens / states and the calls' own (the v2 encoder's inputs). v3 references
    (with extras) are ranked by their extras' ranking and carry the V3_SCALAR_NAMES extras.
    """
    G = len(frames)
    actions = torch.zeros(G, max_refs, *demos.actions[episode].shape[1:])
    scalars = torch.zeros(G, max_refs, n_scalars)
    success = torch.zeros(G, max_refs, dtype=torch.long)
    valid = torch.zeros(G, max_refs, dtype=torch.bool)
    out = {"actions": actions, "scalars": scalars, "success": success, "valid": valid}
    if features:
        (V, Dv), Ds = demos.feature_shapes()
        calls = torch.as_tensor(np.asarray(frames, dtype=np.int64))
        out.update(ref_vis=torch.zeros(G, max_refs, V, Dv, dtype=torch.half), ref_state=torch.zeros(G, max_refs, Ds),
                   cur_vis=demos.vis[episode][calls].clone(), cur_state=demos.state[episode][calls].clone())
    if refs is None:
        return out
    members, ends, sims, costs, m = refs[:5]
    noise = refs[5] if len(refs) > 5 else {}  # candidate column -> chunk noise sigma (near-self references)
    extras = refs[6] if len(refs) > 6 else None  # v3
    order = extras["rank"] if extras is not None else costs
    for g, f in enumerate(frames):
        for slot, c in enumerate(torch.argsort(order[f])[: min(m, max_refs)].tolist()):
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
            values = reference_scalars(j, sim, float(costs[f, c]), int(f), T_r)
            if extras is not None:
                scost = float(extras["state_cost"][f, c])
                if unaligned:
                    scost = float(state_cost(demos.state[r][j], demos.state[episode][int(f)], extras["sigma"]))
                values += [float(extras["task_sim"][c]), extras["text_known"], scost]
            assert len(values) == n_scalars, f"{len(values)} scalars vs the adapter's {n_scalars}"
            scalars[g, slot] = torch.tensor(values)
            success[g, slot] = int(demos.success[r])  # demos: 1; v7 rollouts: their outcome
            valid[g, slot] = True
            if features:
                out["ref_vis"][g, slot], out["ref_state"][g, slot] = demos.vis[r][j], demos.state[r][j]
    return out


def sample_frames(T, group_size, rng):
    if T >= group_size:
        return np.sort(rng.choice(T, group_size, replace=False))
    return np.concatenate([np.arange(T), np.full(group_size - T, T - 1)])  # pad with the last call


# training conditions whose references are wrong by construction (--anchor_bad: these samples must equal the host)
BAD_CONDITIONS = ("other_task", "unaligned")


def make_plan(demos, episodes, refs, conditions, frames, rng, args, negs=None, targets=None, alts=None):
    """
    Dataset indices and adapter inputs of the calls `frames[k]` of `episodes[k]` reading `refs[k]`; with `negs`
    (per episode None or (refs, unaligned)), also the negative adapter inputs and the per-call adherence mask. v9
    `targets`: per episode None (its own executed chunks) or [len(frames[k]), chunk, A] chunks to learn instead (a
    corrective sample: a failure's calls with the aligned success's actions); plan["corrective"] marks those rows.
    v9 v2 `alts`: per episode None or (chunks [len(frames[k]), chunk, A], mask [len(frames[k])]) - a recovery sample's
    contrast (the failed branch's chunks); plan["recovery"] marks the recovery rows, plan["rec_contrast"] the rows whose
    contrast term applies (a chunk exists and the sample reads real references).
    """
    indices, exp_parts, neg_parts, adhere, bad, corrective = [], [], [], [], [], []
    recovery, contrast, alt_chunks = [], [], []
    negs = negs or [None] * len(episodes)
    targets = targets or [None] * len(episodes)
    alts = alts or [False] * len(episodes)  # False: not a recovery sample; None: one without a failed branch
    features = args.encoder == "v2"
    n_scalars = len(V3_SCALAR_NAMES if args.retrieval == "v3" else SCALAR_NAMES)
    # v2: only the slots a sample can fill (<= num_refs); empty slots are masked, so this equals max_refs slots
    slots = args.num_refs if features else args.max_refs
    for episode, ref, condition, frame, neg in zip(episodes, refs, conditions, frames, negs):
        exp_parts.append(reference_batch(demos, episode, frame, ref, condition == "unaligned", rng, slots, features,
                                         n_scalars))
        neg_parts.append(reference_batch(demos, episode, frame, None if neg is None else neg[0],
                                         neg is not None and neg[1], rng, slots, features, n_scalars))
        adhere += [neg is not None] * len(frame)
        bad += [condition in BAD_CONDITIONS] * len(frame)
        indices.extend(demos.indices[episode][int(f)] for f in frame)
    for frame, target in zip(frames, targets):
        corrective += [target is not None] * len(frame)
    cat = lambda parts: {key: torch.cat([p[key] for p in parts]) for key in parts[0]}  # noqa: E731
    chunks = [demos.actions[e][torch.as_tensor(f)] if t is None else t.to(demos.actions[e].dtype)
              for e, f, t in zip(episodes, frames, targets)]
    for chunk, alt, condition in zip(chunks, alts, conditions):
        recovery += [alt is not False] * len(chunk)
        if alt:
            alt_chunks.append(alt[0].to(chunk.dtype))
            contrast += (alt[1] & (condition == args.ref_content)).tolist()
        else:
            alt_chunks.append(chunk)  # placeholder, never scored
            contrast += [False] * len(chunk)
    return {"indices": indices, "exp": cat(exp_parts), "neg_exp": cat(neg_parts), "adhere": torch.tensor(adhere),
            "conditions": conditions, "bad": torch.tensor(bad), "corrective": torch.tensor(corrective),
            "chunks": torch.cat(chunks), "recovery": torch.tensor(recovery, dtype=torch.bool),
            "rec_contrast": torch.tensor(contrast, dtype=torch.bool), "rec_alt": torch.cat(alt_chunks)}


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


def prepare(raw, preprocessor, prompts: PromptMap, chunks=None):
    """
    The policy preprocessor's batch of dataset frames. v7: recorded rollout frames carry no action; their targets are
    the rollouts' executed (normalized) chunks, `chunks` [B, chunk, A] (the plan's).
    """
    if prompts:
        raw = dict(raw)
        raw["task"] = [prompts(t) for t in raw["task"]]
    batch = preprocessor(raw)
    if batch.get(ACTION) is None:  # the preprocessor keeps the key (None) when the frames have no action
        assert chunks is not None, "the frames carry no action and no target chunks were given"
        batch[ACTION] = chunks.to(batch[OBS_STATE].device)
    return batch


def split_episodes(demos, args, rng):
    """
    (training targets, held-out targets, reference universe). Demos: --val_frac of them held out (the first draw of
    `rng`); targets and references are the rest. v7 rollouts: whole variants are held out (the first draw permutes the
    variants), so no attempt of a held-out variant trains; the targets are the successful attempts, the references
    the training variants' attempts (failed ones too with --ref_failures 1).
    """
    if not demos.rollouts:
        permutation = rng.permutation(len(demos)).tolist()
        val = permutation[: max(1, int(len(permutation) * args.val_frac))]
        held = set(val)
        train = [i for i in permutation if i not in held]
        return train, val, train
    variants = sorted(set(demos.variant))
    order = rng.permutation(len(variants)).tolist()
    n_held = max(1, int(len(variants) * args.val_frac))
    if getattr(args, "val_stratify", 0):
        # every stratum (category, difficulty, outcome: all succeeded / mixed / all failed) held out in proportion:
        # variants sorted by stratum (random order within), then every (1 / val_frac)-th one from a random offset
        kind, stratum = variant_outcomes(demos), {}
        for i, v in enumerate(demos.variant):
            stratum[v] = (str(demos.stratum[i]), kind[v])
        rank = {variants[k]: r for r, k in enumerate(order)}
        ranked = sorted(variants, key=lambda v: (stratum[v], rank[v]))
        stride = len(ranked) / n_held
        offset = float(rng.random()) * stride
        held = {ranked[min(len(ranked) - 1, int(offset + k * stride))] for k in range(n_held)}
        counts = collections.Counter(kind[v] for v in held)
        print(f"[split] stratified by category, difficulty and outcome: {len(held)} held-out variants "
              f"({dict(counts)}) of {len(variants)}", flush=True)
    else:
        held = {variants[k] for k in order[:n_held]}
    own = [i for i in range(len(demos)) if demos.branch[i] is None]  # v9 v2: rendered branches are none of these
    targets = [i for i in own if demos.variant[i] not in held and demos.success[i]]
    val = [i for i in own if demos.variant[i] in held and demos.success[i]]
    universe = [i for i in own if demos.variant[i] not in held and (demos.success[i] or args.ref_failures)]
    demos.held_variants = held  # v9 v2: the recovery samples of these variants are validation only
    return targets, val, universe


def variant_outcomes(demos):
    """v7 rollouts: variant -> "success" (all its attempts succeeded), "mixed" or "fail" (none did); the recorded
    attempts only (v9 v2: not the rendered branches)."""
    outcomes = collections.defaultdict(list)
    for v, s, b in zip(demos.variant, demos.success, demos.branch):
        if b is None:
            outcomes[v].append(s)
    return {v: "success" if all(s) else ("fail" if not any(s) else "mixed") for v, s in outcomes.items()}


def target_weights(demos, train_indices, args):
    """
    Sampling probabilities of the training targets (None: uniform): v7 rollouts with --mixed_weight w draw the
    successes of mixed variants (some attempts failed) w times as often, the attempts a successful reference can turn.
    """
    w = float(getattr(args, "mixed_weight", 1.0))
    if not demos.rollouts or w == 1.0:
        return None
    kind = variant_outcomes(demos)
    weights = np.array([w if kind[demos.variant[i]] == "mixed" else 1.0 for i in train_indices])
    mixed = weights != 1.0
    print(f"[sample] mixed-variant successes: {int(mixed.sum())} of {len(train_indices)} targets "
          f"({100 * mixed.mean():.1f}%), drawn with weight {w}: {100 * weights[mixed].sum() / weights.sum():.1f}% "
          f"of the draws", flush=True)
    return weights / weights.sum()


def corrective_items(demos, variants, args):
    """
    v9: the corrective samples of the given (mixed) variants: ({failure F: [(F's call j, success S, S's call t)]},
    success-failure pairs, pairs that part). Every success S is PMA-aligned (robot states in the cost, as the
    retrieval aligns) to its variant's failed attempts; a pair parts at the first call p where F's aligned state is
    farther than --corr_thr (mean squared difference of the normalized states) at p and p + 1, and the --corr_window
    calls t before p give F's aligned calls j (the states still close, the actions already apart); the label is S's
    call t (--corr_label parting) or S's step at F's progress (same_phase: F's call j aligned back to S).
    """
    by_variant = collections.defaultdict(lambda: ([], []))
    for i, (v, s) in enumerate(zip(demos.variant, demos.success)):
        if v in variants and demos.branch[i] is None:
            by_variant[v][0 if s else 1].append(i)
    same_phase = getattr(args, "corr_label", "parting") == "same_phase"
    items, n_pairs, n_parted = collections.defaultdict(list), 0, 0
    for v, (succ, fail) in sorted(by_variant.items()):
        if not succ or not fail:
            continue
        for s in succ:
            ends = align(demos, s, fail, args)[0]  # [T_s, len(fail)]
            for c, f in enumerate(fail):
                n_pairs += 1
                d = ((demos.state[s].float() - demos.state[f][ends[:, c]].float()) ** 2).mean(dim=1)
                far = d > args.corr_thr
                hits = torch.nonzero(far[:-1] & far[1:])[:, 0]
                if not len(hits):
                    continue
                n_parted += 1
                p = int(hits[0])
                # --corr_label same_phase: the label is S's step at F's own progress (F aligned to S, the direction
                # retrieval aligns), not S's call t: aligned from S, t is ahead of F's progress in ~2/3 of the items
                # (median +1 call, F lags), and a delta-pose action from a later phase is wrong at F's position
                back = align(demos, f, [s], args)[0][:, 0] if same_phase else None  # [T_f]
                seen = set()
                for t in range(max(0, p - args.corr_window), p):
                    j = int(ends[t, c])
                    label = int(back[j]) if same_phase else t
                    if (j, label) not in seen:
                        seen.add((j, label))
                        items[f].append((j, s, label))
    return dict(items), n_pairs, n_parted


def branch_items(demos, variants, args):
    """
    v9 --corr_label branch: the corrective samples of the given variants from counterfactual branching
    (branch_collect.py files --branches/--branch_glob): {failure F: [(branch call j, -1, label chunk)]}, one item per
    successful branch (its executed normalized actions from j on, cut / padded with the last to the chunk length), and
    counts. A point whose replay or recorded state at j disagrees with F's (another attempt of the same name) is dropped.
    """
    import glob
    by_name = {n: i for i, n in enumerate(demos.names)}
    items, counts = collections.defaultdict(list), collections.Counter()
    files = sorted(glob.glob(os.path.join(args.branches, args.branch_glob)))
    for path in files:
        for r in torch.load(path, map_location="cpu", weights_only=False):
            e = by_name.get(r["name"])
            if e is None or demos.variant[e] not in variants or demos.success[e]:
                continue
            counts["points"] += 1
            j, C = int(r["branch_call"]), demos.actions[e].shape[1]
            first = r["branches"][0]["states"]
            if (r["replay_state_diff"] > 1e-3 or not len(first) or j >= len(demos.indices[e])
                    or float((first[0] - demos.state[e][j].float()).abs().max()) > 1e-3):
                counts["mismatched"] += 1
                continue
            ok = [b for b in r["branches"] if b["success"] and len(b["actions"])]
            counts["branches"] += len(r["branches"])
            counts["succeeded"] += len(ok)
            counts["labeled"] += bool(ok)
            for b in ok:
                a = b["actions"][:C].float()
                if len(a) < C:
                    a = torch.cat([a, a[-1:].expand(C - len(a), -1)])
                assert a.shape == demos.actions[e].shape[1:], f"{r['name']}: label {tuple(a.shape)} vs chunks"
                items[e].append((j, -1, a))
    counts["files"] = len(files)
    return dict(items), counts


def item_label(demos, other, t):
    """A corrective item's target chunk: the other episode's call t, or (branch items, other -1) the chunk t itself."""
    return t if other < 0 else demos.actions[other][t]


def recovery_items(demos, variants, args):
    """
    v9 v2 --recovery: the recovery samples of the given variants, from the rendered counterfactual branches (Demos'
    branch episodes: F's calls 0..j-1, then the branch's calls j.. on its own observations). Every successful branch
    is a demonstration of how the base policy got out of F's state: its calls j.. (never the shared prefix) with its
    executed chunks as targets, drawn with position weights - 1 at d = call - j in [lo, hi) (--rec_window; the branches
    of a point part a median 8 calls after j, 90 %: 20) except the last --rec_tail calls before the success, else
    --rec_out_weight; each branch point weighs 1 in total (a point where 1 of 5 branches succeeded gives its only
    success the weight of five of an easy point's). The point's rendered failed branch (if any) gives the contrast
    chunks. Returns ({successful branch episode: (calls, weights, failed branch episode or -1, episode weight)},
    counts).
    """
    points = collections.defaultdict(lambda: ([], []))
    for i, b in enumerate(demos.branch):
        if b is not None and demos.variant[i] in variants:
            points[b][0 if demos.success[i] else 1].append(i)
    lo, hi = args.rec_window
    items, counts = {}, collections.Counter()
    for (_, j), (succ, fail) in sorted(points.items()):
        counts["points"] += 1
        counts["no_success"] += not succ
        counts["with_failure"] += bool(succ and fail)
        for e in succ:
            T = demos.actions[e].shape[0]
            d = np.arange(T - j)
            w = np.where((d >= lo) & (d < hi) & (d < T - j - args.rec_tail), 1.0, args.rec_out_weight)
            items[e] = (np.arange(j, T), w, fail[0] if fail else -1, 1.0 / len(succ))
            counts["branches"] += 1
            counts["calls"] += T - j
            counts["window_calls"] += int((w == 1.0).sum())
    return items, counts


def recovery_contrast(demos, alt, frame):
    """The failed branch's chunks at the calls `frame` and where it has them ((chunks, mask); None without one)."""
    if alt < 0:
        return None
    T = demos.actions[alt].shape[0]
    calls = torch.as_tensor(np.minimum(frame, T - 1))
    return demos.actions[alt][calls], torch.as_tensor(frame < T)


def recovery_val_items(demos, items, args):
    """
    v9 v2: the recovery test's items ({successful branch episode: [(call, failed branch episode, call)]}): of held-out
    branch points with both outcomes rendered, one successful branch each (at most --rec_val_points points) and up to
    --rec_val_calls of its window calls where the failed branch has a call too (seeded, fixed for the run).
    """
    rng = np.random.default_rng(args.seed + 7)
    by_point = collections.defaultdict(list)
    for e, (_, _, alt, _) in items.items():
        if alt >= 0:
            by_point[demos.branch[e]].append(e)
    points, out = sorted(by_point), {}
    for k in rng.permutation(len(points))[: args.rec_val_points]:
        its = sorted(by_point[points[k]])
        e = its[int(rng.integers(len(its)))]
        calls, w, alt, _ = items[e]
        ok = calls[(w == 1.0) & (calls < demos.actions[alt].shape[0])]
        if len(ok):
            chosen = np.sort(rng.choice(ok, size=min(len(ok), args.rec_val_calls), replace=False))
            out[e] = [(int(c), alt, int(c)) for c in chosen]
    return out


def load_training_data(policy, args):
    """
    (store, Demos, dataset factory). Demos: the library's demos, frames from its LeRobot dataset. v7 (--rollouts): the
    rollouts recorded in those directories (eval_suite --record_frames) and their frames; the library is loaded too,
    as the evaluation loads it: the rollouts' keys must use its key transform, and its tasks fit the instruction index.
    """
    if not getattr(args, "rollouts", None):
        store = ExperienceStore.load(args.library)
        return store, Demos(store, args.max_episodes), lambda: load_dataset(
            policy, args.dataset_repo_id, args.dataset_root, args.video_backend, parse_camera_map(args.camera_map))
    dirs = list(args.rollouts) + list(getattr(args, "recovery", None) or [])  # v9 v2: + the rendered branches
    files, frames = rollout_store_files(dirs), frame_files(dirs)
    if not files or not frames:
        raise SystemExit(f"--rollouts {args.rollouts}: no rollouts_*.pt / frames_*.pt (record them with "
                         "pace.eval_suite --record_frames)")
    store = ExperienceStore.load([args.library] + [str(f) for f in files])  # refuses rollouts keyed otherwise
    first = [RolloutFrames(frames)]
    print(f"[rollouts] {len(files)} rollout files, {len(frames)} frame files, {len(first[0])} recorded calls",
          flush=True)
    demos = Demos(store, args.max_episodes, frames=first[0])
    return store, demos, lambda: first.pop() if first else RolloutFrames(frames)


def instruction_sims(policy, store, tasks):
    """
    v3: instruction similarities [K, K] of the sorted `tasks`. Demos: the index fitted on them (as before); v7
    rollouts: fitted on the library's demo tasks, as the evaluation fits it (`pi05.PI05Experience`).
    """
    fit = sorted({ep["task_key"] for ep in store.episodes if ep["source"] == "demo"})
    if not fit or all(ep["source"] == "demo" for ep in store.episodes):
        fit = tasks
    text = InstructionText.from_policy(policy, fit)
    embedded = torch.stack([text.embed(t) for t in tasks])
    return embedded @ embedded.T


def check_chunks(batch, plan, step):
    """The dataset chunks (and, with features, robot states) must be the library's (same data, normalization, frames)."""
    got = batch[ACTION].float().cpu()
    err = float((got - plan["chunks"]).abs().max())
    if err > 1e-3:
        raise RuntimeError(f"step {step}: dataset action chunks differ from the library's by {err:.4g}; was the "
                           "library built from this dataset with this policy's preprocessor?")
    if "cur_state" in plan["exp"]:
        err = float((batch_state(batch).cpu() - plan["exp"]["cur_state"]).abs().max())
        if err > 1e-3:
            raise RuntimeError(f"step {step}: dataset robot states differ from the library's by {err:.4g}")


def scored_loss(policy, pace, batch, exp, repeats, noise=None, time=None, per_step=False):
    """
    Mean over --repeats draws of the per-sample loss [B] ([B, chunk] per action step with `per_step`); pass noise/time
    [R, ...] for identical draws.
    """
    prefix = prefix_cache(policy, batch)
    actions = batch[ACTION].float()
    pace.set_exp(exp)
    try:
        losses = [flow_loss(policy, prefix, actions, noise=None if noise is None else noise[r],
                            time=None if time is None else time[r], per_step=per_step)
                  for r in range(repeats)]
    finally:
        pace.set_exp(None)
    return torch.stack(losses).mean(dim=0)


def take(value, rows, B):
    """`value[rows]` for a tensor / list with one entry per sample (leading size B); anything else unchanged."""
    if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == B:
        return value[rows]
    if isinstance(value, (list, tuple)) and len(value) == B:
        return type(value)(value[i] for i in rows.tolist()) if torch.is_tensor(rows) else value[rows]
    return value


def training_loss(policy, pace, batch, plan, args):
    """
    BC loss (+ the --adhere term) of one training batch; returns (detached scalar loss, whether any gradient reached
    the adapter, stats, v6 relevance stats of the batch's own references (empty without a gate)). The loss is back-propagated one (noise, time) draw at a time, and with --micro_batches k > 1
    the samples are also processed in k consecutive slices (each with its own prefix cache and draws), so only one
    slice's graphs are alive. Every term is a sum over samples divided by the whole batch's count (batch size, number
    of samples with an adherence term), so the gradient is that of the whole batch whatever k is.
    """
    actions_all = batch[ACTION].float()
    B, device = actions_all.shape[0], actions_all.device
    head = int(getattr(args, "loss_steps", 0) or 0)

    def chunk_loss(prefix, actions, noise, time_):
        """[B]: the flow-matching loss over the whole chunk, or (--loss_steps H) over its first H steps."""
        if not head:
            return flow_loss(policy, prefix, actions, noise=noise, time=time_)
        return flow_loss(policy, prefix, actions, noise=noise, time=time_, per_step=True)[:, :head].mean(dim=1)

    mask_all = plan["adhere"].to(device).float()
    corr_all = plan["corrective"].to(device) if "corrective" in plan else None  # v9: corrective rows (stats only)
    n_corr = int(corr_all.sum()) if corr_all is not None else 0
    # v9 v2: recovery rows (stats) and the rows of the contrast term (--rec_contrast)
    rec_all = plan["recovery"].to(device) if "recovery" in plan else None
    n_rec = int(rec_all.sum()) if rec_all is not None else 0
    contrast_w = float(getattr(args, "rec_contrast", 0.0) or 0.0)
    con_all = plan["rec_contrast"].to(device) if contrast_w and "rec_contrast" in plan else None
    n_con = int(con_all.sum()) if con_all is not None else 0
    relative_corr = bool(getattr(args, "corr_relative", 0))
    anchor = float(getattr(args, "anchor_bad", 0.0) or 0.0)
    bad_all = plan["bad"].to(device) if anchor and "bad" in plan else None
    symmetric = bool(getattr(args, "adhere_symmetric", 0))
    use_all = args.adhere != "none" and bool(mask_all.any())
    n_total = mask_all.sum().clamp(min=1.0)
    eps = 1e-6
    total, stats, backward = 0.0, collections.defaultdict(float), False
    relevance, n_relevance = collections.defaultdict(float), 0  # v6: of the samples' own references
    k = max(1, min(int(args.micro_batches), B))
    for rows in torch.arange(B).chunk(k):
        whole = len(rows) == B
        sub = batch if whole else {key: take(v, rows, B) for key, v in batch.items()}
        exp = plan["exp"] if whole else {key: take(v, rows, B) for key, v in plan["exp"].items()}
        neg_exp = plan["neg_exp"] if whole else {key: take(v, rows, B) for key, v in plan["neg_exp"].items()}
        actions, mask = sub[ACTION].float(), mask_all[rows.to(device)]
        bad = bad_all[rows.to(device)] if bad_all is not None else None
        use = use_all and bool(mask.any())
        prefix = prefix_cache(policy, sub)
        for _ in range(args.repeats):
            noise = policy.model.sample_noise((len(rows), policy.config.chunk_size, policy.config.max_action_dim), device)
            time_ = policy.model.sample_time(len(rows), device)
            pace.adapter.relevance_stats = {}
            pace.set_exp(exp)
            if pace.adapter.relevance_stats:
                for name, value in pace.adapter.relevance_stats.items():
                    relevance[name] += value
                n_relevance += 1
            l_real = chunk_loss(prefix, actions, noise, time_)
            per_sample = l_real
            corr = corr_all[rows.to(device)] if n_corr else None
            has_bad = bad is not None and bool(bad.any())
            relative = corr is not None and bool(corr.any()) and relative_corr
            l_host = None
            if has_bad or relative:
                pace.set_exp(None)
                with torch.no_grad():
                    l_host = chunk_loss(prefix, actions, noise, time_)
            if has_bad:
                # v8 --anchor_bad: a sample read with wrong references (other task, misaligned) must leave the host's
                # loss unchanged in EITHER direction: a gain there is a correction that does not depend on the
                # reference (v7 learned one), not the use of experience
                shift = l_real - l_host
                per_sample = torch.where(bad, anchor * shift.abs(), l_real)
                stats["bad_shift"] += float(shift.detach()[bad].sum()) / B / args.repeats
                stats["bad_abs_shift"] += float(shift.detach().abs()[bad].sum()) / B / args.repeats
            if corr is not None:
                stats["corr_l_real"] += float(l_real.detach()[corr].sum()) / n_corr / args.repeats
            if relative:
                # v9 --corr_relative: a corrective sample's term (its loss, or its anchor) relative to the host's loss
                # on it, times --corr_scale (the plain samples' loss level): the host is far from the success's action
                # at a failure's observation (~25x a plain sample's loss), which would hand most of the gradient to
                # the few corrective calls; this way each weighs as a plain sample and counts the fraction it improves
                scale = args.corr_scale / l_host.clamp(min=1e-6)
                per_sample = torch.where(corr, per_sample * scale, per_sample)
                stats["corr_rel"] += float((l_real.detach() / l_host.clamp(min=1e-6))[corr].sum()) / n_corr / args.repeats
            loss = per_sample.sum() / B
            stats["l_real"] += float(l_real.detach().sum()) / B / args.repeats
            if n_rec:
                rec = rec_all[rows.to(device)]
                stats["rec_l_real"] += float(l_real.detach()[rec].sum()) / n_rec / args.repeats
            con = con_all[rows.to(device)] if n_con else None
            if con is not None and bool(con.any()):
                # v9 v2 contrast: at a successful branch's observation, the adapter must prefer its action over the
                # failed branch's (same call after the branch point) by a relative margin beyond the host's own
                # preference: relu(m - [(log L_alt - log L_own)_adapter - (log L_alt - log L_own)_host]); gradients
                # through both losses, bounded by the margin (no unlimited push of the failed action's loss)
                alt = plan["rec_alt"][rows].to(device, actions.dtype)
                pace.set_exp(exp)
                l_alt = chunk_loss(prefix, alt, noise, time_)
                pace.set_exp(None)
                with torch.no_grad():
                    l_alt_host = chunk_loss(prefix, alt, noise, time_)
                    if l_host is None:
                        l_host = chunk_loss(prefix, actions, noise, time_)
                gap = (torch.log(l_alt + eps) - torch.log(l_real + eps))                     - (torch.log(l_alt_host + eps) - torch.log(l_host + eps))
                term = F.relu(args.rec_margin - gap)
                loss = loss + contrast_w * term[con].sum() / B
                stats["rec_gap"] += float(gap.detach()[con].sum()) / n_con / args.repeats
                stats["rec_contrast_active"] += float((term.detach()[con] > 0).float().sum()) / n_con / args.repeats
            if use:
                pace.set_exp(neg_exp)
                l_neg = chunk_loss(prefix, actions, noise, time_)
                n = n_total
                if args.adhere == "pace":
                    pace.set_exp(None)
                    with torch.no_grad():
                        l_base = chunk_loss(prefix, actions, noise, time_)
                    rank = F.relu(torch.log(l_real + eps) - torch.log(l_neg.detach() + eps) + args.adhere_margin)
                    harm = torch.log(l_neg + eps) - torch.log(l_base + eps)
                    # v8 --adhere_symmetric: the negative must not help either (|.|), not only not hurt (relu)
                    harm = harm.abs() if symmetric else F.relu(harm)
                    loss = loss + (args.adhere_rank * rank + args.adhere_harm * harm).mul(mask).sum() / n
                    stats["rank_active"] += float(((rank > 0).float() * mask).sum() / n) / args.repeats
                    stats["harm"] += float((harm.detach() * mask).sum() / n) / args.repeats
                    stats["log_real_over_base"] += float(
                        ((torch.log(l_real.detach() + eps) - torch.log(l_base + eps)) * mask).sum() / n) / args.repeats
                else:  # margin (RA-VLA)
                    term = F.relu(args.adhere_abs_margin - (l_neg - l_real))
                    loss = loss + args.adhere_lambda * term.mul(mask).sum() / n
                    stats["margin_active"] += float(((term > 0).float() * mask).sum() / n) / args.repeats
                stats["log_real_over_neg"] += float(
                    ((torch.log(l_real.detach() + eps) - torch.log(l_neg.detach() + eps)) * mask).sum() / n) / args.repeats
            pace.set_exp(None)
            if loss.requires_grad:  # a batch without any reference has no path to the adapter
                (loss / args.repeats).backward()
                backward = True
            total += float(loss.detach()) / args.repeats
    relevance = {name: value / n_relevance for name, value in relevance.items()}
    return total, backward, stats, relevance


def build_validation(demos, alignments, val_indices, dataset, args):
    """
    Fixed held-out batches: identical calls and (noise, time) draws in every condition; aligned and unaligned read
    the same references (the --num_refs best-aligned members of the full pool at every call).
    """
    rng = np.random.default_rng(args.seed + 1)
    sparse_rng = np.random.default_rng(args.seed + 3)
    chosen = [val_indices[i] for i in rng.permutation(len(val_indices))[: args.val_episodes]]
    batches = []
    group = args.val_group or args.episodes_per_step  # the draws depend on it: keep it equal across compared runs
    for start in range(0, len(chosen), group):
        episodes = chosen[start : start + group]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, rng) for e in episodes]
        text = {"text": True} if alignments.v3 else {}  # v3: validated with the instruction, and without (notext)
        keep = {"variant": True} if alignments.variants is not None else {}  # v7: the own variant's attempts stay
        refs = {c: [alignments.references(e, c, rng, num=args.num_refs, full_pool=True, **text, **keep)
                    for e in episodes]
                for c in ("real", "other_task")}
        if alignments.v3:
            refs["notext"] = [alignments.references(e, "real", rng, num=args.num_refs, full_pool=True, text=False,
                                                    **keep)
                              for e in episodes]
        if args.ref_content == "self":
            refs["self"] = [alignments.references(e, "self", rng, **text) for e in episodes]
        if alignments.v3 and getattr(args, "p_sparse", 0.0) > 0:  # own stream: the other conditions keep their draws
            refs["sparse"] = [alignments.references(e, "sparse", sparse_rng, num=args.num_refs, text=True)
                              for e in episodes]
        if alignments.variants is not None:  # v7: real keeps the own variant's other attempts, novariant drops them
            refs["novariant"] = [alignments.references(e, "real", sparse_rng, num=args.num_refs, full_pool=True,
                                                       text=True, variant=False) for e in episodes]
        conditions = {}
        for condition in ("real", "notext", "unaligned", "other_task", "none", "self", "sparse", "novariant"):
            if (condition == "other_task" and not alignments.has_other_tasks) or \
                    (condition in ("self", "notext", "sparse", "novariant") and condition not in refs):
                continue
            source = refs[condition] if condition in ("other_task", "self", "notext", "sparse", "novariant") \
                else refs["real"]
            conditions[condition] = make_plan(demos, episodes, source, [condition] * len(episodes), frames, rng,
                                              args)
        first = next(iter(conditions.values()))
        raw = default_collate([dataset[i] for i in first["indices"]])
        batches.append({"conditions": conditions, "raw": raw, "E": len(episodes), "chunks": first["chunks"],
                        "episodes": episodes})
    return batches


@torch.no_grad()
def validate(policy, pace, batches, preprocessor, prompts, args, mixed=None):
    """
    Held-out losses per condition: val_<c> over the whole chunk, val10_<c> over its first n_action_steps steps (what
    the closed loop executes of a call's prediction; the rest of a rollout's executed chunk came from later calls).
    v7 `mixed` (episode indices of mixed variants): also val10_mixed_<c> on those episodes.
    """
    pace.adapter.eval()
    head = args.n_action_steps or policy.config.n_action_steps
    losses = collections.defaultdict(list)
    losses10 = collections.defaultdict(list)
    is_mixed = []
    relevance = collections.defaultdict(list)  # v6: mean reference weight g per condition
    refmass = collections.defaultdict(list)  # v6: the input injection's total weight on the references
    # The same (noise, time) draws in every condition and every validation pass; the training RNG is left untouched
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed + 2)
        for batch_data in batches:
            batch = prepare(batch_data["raw"], preprocessor, prompts, batch_data["chunks"])
            B = batch[ACTION].shape[0]
            shape = (args.repeats, B, policy.config.chunk_size, policy.config.max_action_dim)
            noise = torch.randn(shape, device="cuda")
            time_ = policy.model.sample_time(args.repeats * B, "cuda").view(args.repeats, B)
            if mixed is not None:
                is_mixed += [e in mixed for e in batch_data["episodes"]]
            for condition, plan in batch_data["conditions"].items():
                pace.adapter.relevance_stats = {}
                steps = scored_loss(policy, pace, batch, None if condition == "none" else plan["exp"], args.repeats,
                                    noise=noise, time=time_, per_step=True)
                loss, loss10 = steps.mean(dim=1), steps[:, :head].mean(dim=1)
                losses[condition].append(loss.view(batch_data["E"], args.group_size).mean(dim=1).cpu())
                losses10[condition].append(loss10.view(batch_data["E"], args.group_size).mean(dim=1).cpu())
                if "relevance_mean" in pace.adapter.relevance_stats:
                    relevance[condition].append(pace.adapter.relevance_stats["relevance_mean"])
                if "inject_ref_mass" in pace.adapter.relevance_stats:
                    refmass[condition].append(pace.adapter.relevance_stats["inject_ref_mass"])
    pace.adapter.train()
    per_episode = {c: torch.cat(v).double() for c, v in losses.items()}
    per_episode10 = {c: torch.cat(v).double() for c, v in losses10.items()}
    result = {f"val_{c}": float(v.mean()) for c, v in per_episode.items()}
    result.update({f"val10_{c}": float(v.mean()) for c, v in per_episode10.items()})
    sets = [("", per_episode), ("10", per_episode10)]
    if is_mixed and any(is_mixed):
        rows = torch.tensor(is_mixed)
        mixed10 = {c: v[rows] for c, v in per_episode10.items()}
        result.update({f"val10_mixed_{c}": float(v.mean()) for c, v in mixed10.items()})
        result["val_mixed_episodes"] = int(rows.sum())
        sets.append(("10_mixed", mixed10))
    result.update({f"relevance_{c}": float(np.mean(v)) for c, v in relevance.items()})
    result.update({f"refmass_{c}": float(np.mean(v)) for c, v in refmass.items()})
    for suffix, table in sets:
        for base, other in (("real", "none"), ("real", "unaligned"), ("real", "other_task"), ("self", "none"),
                            ("real", "notext"), ("notext", "none"), ("sparse", "none"), ("unaligned", "none"),
                            ("other_task", "none"), ("real", "novariant"), ("novariant", "none")):
            if suffix == "10_mixed" and (base, other) not in (("real", "none"), ("real", "novariant"),
                                                              ("novariant", "none")):
                continue
            if base in table and other in table:
                diff = table[base] - table[other]
                name = f"{base}_minus_{other}" + (f"_{suffix}" if suffix else "")
                result[name] = float(diff.mean())
                result[f"{name}_se"] = float(diff.std() / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return result


CORR_CONDITIONS = ("real", "novariant", "other_task", "unaligned", "none")


def build_corrective_validation(demos, alignments, items, dataset, args):
    """
    v9: the corrective test's fixed batches, one per observed episode of `items` ({episode: [(its call, other
    episode, that one's aligned call)]}: failures with their aligned successes, or `success_side` of them): its calls
    read with the references of train.py's validation per condition (real keeps the own variant's attempts,
    novariant drops them, other_task, unaligned = real at random steps, none), its own chunks and the other's.
    """
    rng = np.random.default_rng(args.seed + 5)
    batches = []
    for e, its in sorted(items.items()):
        frame = np.array([j for j, _, _ in its])
        refs = {
            "real": alignments.references(e, "real", rng, num=args.num_refs, full_pool=True, text=True, variant=True),
            "novariant": alignments.references(e, "real", rng, num=args.num_refs, full_pool=True, text=True,
                                               variant=False),
            "other_task": alignments.references(e, "other_task", rng, num=args.num_refs, full_pool=True, text=True,
                                                variant=True),
        }
        conditions = {c: make_plan(demos, [e], [refs[c if c in refs else "real"]], [c], [frame], rng, args)
                      for c in CORR_CONDITIONS}
        batches.append({"conditions": conditions, "own": conditions["real"]["chunks"],
                        "alt": torch.stack([item_label(demos, o, t).to(demos.actions[e].dtype) for _, o, t in its]),
                        "raw": default_collate([dataset[i] for i in conditions["real"]["indices"]])})
    return batches


def success_side(items):
    """The corrective items seen from the successes: {S: [(S's call t, F, F's aligned call j)]}."""
    out = collections.defaultdict(list)
    for f, its in items.items():
        for j, s, t in its:
            out[s].append((t, f, j))
    return dict(out)


@torch.no_grad()
def validate_corrective(policy, pace, batches, preprocessor, prompts, args, prefix="val_corr", own="F", alt="S",
                        toward_alt=True):
    """
    v9 corrective test (prefix val_corr): at a held-out failure's observation, the loss (first n_action_steps steps)
    of its own action F and of the aligned success's action S per condition, identical (noise, time) draws; the base
    prefers F there (it produced it), and <prefix>_<c>_gain = (F - S)_c - (F - S)_none > 0 when the references pull
    the prediction toward the success's way. Success side (prefix val_succ, own S, alt F, toward_alt False): at the
    success's observation the base prefers S; gain = (F - S)_c - (F - S)_none must not turn negative (a correction
    that also pulls successes off their way). se clustered by observed episode.
    """
    pace.adapter.eval()
    head = args.n_action_steps or policy.config.n_action_steps
    loss_own, loss_alt, sizes = collections.defaultdict(list), collections.defaultdict(list), []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(args.seed + (6 if toward_alt else 8))
        for b in batches:
            batch_own = prepare(b["raw"], preprocessor, prompts, b["own"])
            batch_alt = dict(batch_own)
            batch_alt[ACTION] = b["alt"].to(batch_own[ACTION].device, batch_own[ACTION].dtype)
            B = batch_own[ACTION].shape[0]
            noise = torch.randn((args.repeats, B, policy.config.chunk_size, policy.config.max_action_dim),
                                device="cuda")
            time_ = policy.model.sample_time(args.repeats * B, "cuda").view(args.repeats, B)
            for c, plan in b["conditions"].items():
                exp = None if c == "none" else plan["exp"]
                for table, batch in ((loss_own, batch_own), (loss_alt, batch_alt)):
                    steps = scored_loss(policy, pace, batch, exp, args.repeats, noise=noise, time=time_, per_step=True)
                    table[c].append(steps[:, :head].mean(dim=1).cpu())
            sizes.append(B)
    pace.adapter.train()
    O = {c: torch.cat(v).double() for c, v in loss_own.items()}
    A = {c: torch.cat(v).double() for c, v in loss_alt.items()}
    owner = torch.cat([torch.full((n,), k) for k, n in enumerate(sizes)])
    sign = 1.0 if toward_alt else -1.0
    out = {f"{prefix}_calls": int(sum(sizes)), f"{prefix}_episodes": len(sizes)}
    for c in O:
        out[f"{prefix}_{c}_{own}"], out[f"{prefix}_{c}_{alt}"] = float(O[c].mean()), float(A[c].mean())
        if c == "none":
            continue
        gain = sign * ((O[c] - A[c]) - (O["none"] - A["none"]))
        per = torch.stack([gain[owner == k].mean() for k in range(len(sizes))])
        out[f"{prefix}_{c}_gain"] = float(gain.mean())
        out[f"{prefix}_{c}_gain_se"] = float(per.std() / math.sqrt(len(per))) if len(per) > 1 else float("nan")
        out[f"{prefix}_{c}_{own}_rel"] = float(O[c].mean() / O["none"].mean() - 1)
        out[f"{prefix}_{c}_{alt}_rel"] = float(A[c].mean() / A["none"].mean() - 1)
    return out


def validate_both_sides(policy, pace, batches, preprocessor, prompts, args):
    """v9: the corrective test at the failures' observations (val_corr_*) and at the successes' (val_succ_*)."""
    out = validate_corrective(policy, pace, batches[0], preprocessor, prompts, args)
    out.update(validate_corrective(policy, pace, batches[1], preprocessor, prompts, args, prefix="val_succ", own="S",
                                   alt="F", toward_alt=False))
    return out


def validate_recovery(policy, pace, batches, preprocessor, prompts, args):
    """
    v9 v2 recovery test (val_rec_*): at held-out successful branches' observations after their branch points, the loss
    of the branch's own action S and of the failed branch's at the same call F per condition; gain = (F - S)_c -
    (F - S)_none > 0 when the references move the prediction toward the way that succeeded, and val_rec_<c>_S_rel the
    relative change of S's loss (real must beat other_task: the help has to come from the experience).
    """
    return validate_corrective(policy, pace, batches, preprocessor, prompts, args, prefix="val_rec", own="S", alt="F",
                               toward_alt=False)


def setup_distributed():
    """(rank, world size) under torchrun (NCCL, one GPU per process; long timeout: the others wait while rank 0 validates)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=3), device_id=torch.device("cuda", local))
    return dist.get_rank(), world


def average_gradients(params, world):
    """Mean of the adapter gradients over the processes (a process without a gradient contributes zeros)."""
    assert len({p.dtype for p in params}) == 1, "mixed parameter dtypes"
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p in params:
        p.grad = flat[offset : offset + p.numel()].view_as(p)
        offset += p.numel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--dataset_repo_id", default=None, help="Demos' dataset (not with --rollouts)")
    parser.add_argument("--rollouts", nargs="*", default=[],
                        help="v7: train on the policy's own rollouts recorded in these directories (pace.eval_suite "
                             "--record_frames: <out_dir>/rollouts) instead of demos; --library is then the evaluation's "
                             "library (key transform, instruction index), and the v3 retrieval is needed")
    parser.add_argument("--ref_failures", type=int, default=0,
                        help="v7: failed attempts are references too (their outcome is an adapter input); the "
                             "evaluation then retrieves failures as well (stored in the adapter)")
    parser.add_argument("--p_same_variant", type=float, default=0.5,
                        help="v7: probability that a sample's other attempts of its own variant stay candidates (a "
                             "closed-loop store holds them from the second round on)")
    parser.add_argument("--ref_isolation", type=int, default=0,
                        help="v8 (v2 encoder): the reference tokens carry the references only - the current call stays "
                             "out of the encoder's context and of the summary (see core.ExperienceAdapter)")
    parser.add_argument("--anchor_bad", type=float, default=0.0,
                        help="v8: samples read with wrong references (other task, misaligned) train on this times "
                             "|loss - host loss| instead of their loss: they must leave the host unchanged in either "
                             "direction (0 = off: they train on their loss like the others)")
    parser.add_argument("--adhere_symmetric", type=int, default=0,
                        help="v8: the --adhere pace harm term penalizes a negative's loss moving away from the host's "
                             "in either direction (|log ratio|) instead of only above it")
    parser.add_argument("--p_corrective", type=float, default=0.0,
                        help="v9 (rollouts): this fraction of each step's episodes are corrective samples - a mixed "
                             "variant's failed attempt's calls just before it parts from a success, with the aligned "
                             "success's actions as targets (corrective_items); also enables the corrective test")
    parser.add_argument("--corr_thr", type=float, default=0.297,
                        help="v9: parting threshold on the aligned robot-state distance (mean squared difference of "
                             "the normalized states; the 95th percentile of success-to-success distances on the v7 "
                             "rollouts, decision_diag.py)")
    parser.add_argument("--corr_window", type=int, default=3, help="v9: calls before the parting call used")
    parser.add_argument("--corr_label", choices=("parting", "same_phase", "branch"), default="parting",
                        help="v9: the corrective target - S's call paired by the parting alignment (S aligned to F; "
                             "ahead of F's progress in ~2/3 of the items), or S's step at F's own progress "
                             "(same_phase: F aligned to S, as retrieval aligns), or the successful counterfactual "
                             "branches' actions (branch: --branches)")
    parser.add_argument("--branches", default="/root/pace/runs/branches_v10",
                        help="v9 --corr_label branch: directory of branch_collect.py's files")
    parser.add_argument("--branch_glob", default="branches_s[0-9]*_*.pt",
                        help="v9 --corr_label branch: which of its files (default: the untagged ones, no smoke)")
    parser.add_argument("--corr_relative", type=int, default=0,
                        help="v9: a corrective sample's loss (or anchor) is divided by the host's loss on it and "
                             "multiplied by --corr_scale, so it weighs as a plain sample (the host's loss there is "
                             "~25x a plain sample's; 0 = absolute: the corrective calls take ~90%% of the gradient)")
    parser.add_argument("--corr_scale", type=float, default=0.04,
                        help="v9 --corr_relative: the plain samples' loss level (first-10-step loss ~0.035-0.045)")
    parser.add_argument("--recovery", nargs="*", default=[],
                        help="v9 v2: directories of rendered counterfactual branches (render_branches.py: "
                             "<dir>/rollouts); their successful branches give recovery samples (recovery_items), "
                             "never targets, references or validation episodes")
    parser.add_argument("--p_recovery", type=float, default=0.0,
                        help="v9 v2: this fraction of each step's episodes are recovery samples")
    parser.add_argument("--rec_window", type=int, nargs=2, default=(2, 24),
                        help="v9 v2: calls d after the branch point drawn with weight 1 (lo <= d < hi)")
    parser.add_argument("--rec_tail", type=int, default=3,
                        help="v9 v2: the last this many calls of a branch (the end of the task) weigh --rec_out_weight")
    parser.add_argument("--rec_out_weight", type=float, default=0.25,
                        help="v9 v2: draw weight of the calls outside the window (> 0)")
    parser.add_argument("--rec_contrast", type=float, default=0.0,
                        help="v9 v2: weight of the contrast term (the successful branch's action preferred over the "
                             "failed branch's, beyond the host's preference, by --rec_margin); 0 = off")
    parser.add_argument("--rec_margin", type=float, default=0.05, help="v9 v2: contrast margin (log ratio)")
    parser.add_argument("--rec_val_points", type=int, default=64,
                        help="v9 v2: held-out branch points of the recovery test (one successful branch each)")
    parser.add_argument("--rec_val_calls", type=int, default=8, help="v9 v2: window calls per recovery-test branch")
    parser.add_argument("--corr_val", type=int, default=0,
                        help="v9: run the corrective test on the held-out mixed variants without corrective training")
    parser.add_argument("--loss_steps", type=int, default=0,
                        help="Train on the flow-matching loss of the first this many steps of each chunk only (0 = the "
                             "whole chunk). v7: --n_action_steps, the steps a call's prediction executes in closed loop; "
                             "the rest of a rollout's executed chunk came from later calls' plans")
    parser.add_argument("--max_same_variant", type=int, default=0,
                        help="v7: when a sample keeps its own variant's attempts, keep 1..this many of them at random "
                             "(a closed-loop run of R rounds stores at most R-1; 0 = all of them)")
    parser.add_argument("--val_stratify", type=int, default=0,
                        help="v7: hold out the --val_frac variants in proportion per stratum (LIBERO-plus category, "
                             "difficulty, and outcome: all attempts succeeded / mixed / none did); 0 = uniformly")
    parser.add_argument("--mixed_weight", type=float, default=1.0,
                        help="v7: draw the successful attempts of mixed variants (some attempts failed) this many "
                             "times as often as the others (1 = uniform over the successful attempts)")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--video_backend", default=None)
    parser.add_argument("--camera_map", nargs="*", default=[],
                        help="<dataset key>=<policy key> renames (lerobot/libero_plus: observation.images.front="
                             "observation.images.image observation.images.wrist=observation.images.image2)")
    parser.add_argument("--library", required=True, help="Demo library of this dataset (pace/build_library.py)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_action_steps", type=int, default=None)
    parser.add_argument("--prompt_override", default=None, help="Tell the policy this instead of every instruction")
    parser.add_argument("--prompt_map", default=None, help="JSON {instruction: prompt} ('*' = default)")
    parser.add_argument("--ref_content", choices=("real", "unaligned", "self"), default="real")
    parser.add_argument("--num_refs", type=int, default=4, help="Maximum references read per call")
    parser.add_argument("--max_refs", type=int, default=8, help="Adapter slots (>= num_refs; deploy may use up to this)")
    parser.add_argument("--vmax", type=int, default=3, help="PMA: largest memory advance per call")
    parser.add_argument("--align", choices=("pma", "nn"), default="pma",
                        help="Reference alignment: PMA over the history, or frame nearest neighbour (as deployed)")
    parser.add_argument("--gamma", type=float, default=0.01,
                        help="PMA: soft prior gamma*|advance-1|, on the scale of the per-call cost 1-cos (see [pma] log)")
    parser.add_argument("--p_none", type=float, default=0.25)
    parser.add_argument("--p_distractor", type=float, default=0.1)
    parser.add_argument("--p_misaligned", type=float, default=0.0,
                        help="v6: probability of a sample whose references are read at random steps (bad references; "
                             "the target is unchanged, so the adapter learns to decline them)")
    parser.add_argument("--p_sparse", type=float, default=0.0,
                        help="v6 (v3 retrieval): probability of a sample whose references come from a sparse store: "
                             "1..--sparse_max random demos, of its own task (other scenes) or of other tasks, aligned as "
                             "deployed (closed loop early on); the target is unchanged")
    parser.add_argument("--sparse_max", type=int, default=6, help="v6: largest sparse store")
    parser.add_argument("--sparse_same_task", type=float, default=0.6,
                        help="v6: probability that a sparse-store demo is of the sample's own task")
    parser.add_argument("--p_subpool", type=float, default=0.5)
    parser.add_argument("--same_scene_threshold", type=float, default=0.999)
    parser.add_argument("--p_near_self", type=float, default=0.0)
    parser.add_argument("--near_self_offset", type=int, default=2)
    parser.add_argument("--near_self_noise", type=float, default=0.2, help="Largest chunk noise sigma (normalized)")
    parser.add_argument("--adhere", choices=("pace", "margin", "none"), default="pace",
                        help="Contrastive term on samples with real references (see the module docstring)")
    parser.add_argument("--adhere_rank", type=float, default=0.1)
    parser.add_argument("--adhere_margin", type=float, default=0.05, help="pace: log margin (~5%% relative)")
    parser.add_argument("--adhere_harm", type=float, default=0.1)
    parser.add_argument("--adhere_lambda", type=float, default=0.1, help="margin: weight (RA-VLA 0.1)")
    parser.add_argument("--adhere_abs_margin", type=float, default=0.01, help="margin: m (RA-VLA 0.01)")
    parser.add_argument("--p_neg_other", type=float, default=0.5,
                        help="Negative = another task's demos with this probability, else the real refs unaligned")
    parser.add_argument("--inject_layers", nargs="+", default=["every2"],
                        help="Expert layers with a cross-attention: indices and/or all, every<N>, last<N>, first<N>, "
                             "last_half, first_half (pace.pi05.resolve_layers)")
    parser.add_argument("--position", choices=("post_layer", "pre_mlp"), default="post_layer",
                        help="Cross-attention after the expert block, or between its joint attention and its MLP")
    parser.add_argument("--time_cond", type=int, default=0,
                        help="1: condition every cross-attention on the flow-time embedding (adaLN, zero-init)")
    parser.add_argument("--input_inject", type=int, default=1)
    parser.add_argument("--encoder", choices=("v1", "v2"), default="v1",
                        help="Reference encoder (core.ExperienceAdapter); v2 needs a library built with --features 1")
    parser.add_argument("--enc_dim", type=int, default=512, help="v2: width of the reference transformer")
    parser.add_argument("--enc_layers", type=int, default=2, help="v2: layers of the reference transformer")
    parser.add_argument("--enc_heads", type=int, default=8, help="v2: heads of the reference transformer")
    parser.add_argument("--act_segments", type=int, default=10, help="v2: action tokens per reference chunk")
    parser.add_argument("--retrieval", choices=("classic", "v3"), default="classic",
                        help="classic: references from the task's own demos, aligned on image keys (v1, v2); v3: robot "
                             "states join the alignment and candidates come from the tasks closest in instruction and "
                             "scene (needs a library with robot states)")
    parser.add_argument("--state_weight", type=float, default=None,
                        help="v3: weight of the robot-state cost in the local alignment cost (default 0.3)")
    parser.add_argument("--state_sigma", type=float, default=0.3, help="v3: scale of the robot-state cost")
    parser.add_argument("--text_weight", type=float, default=0.2,
                        help="v3: ranking penalty text_weight * (1 - instruction similarity of the candidate's task)")
    parser.add_argument("--text_tasks", type=int, default=3, help="v3: candidate tasks closest in instruction")
    parser.add_argument("--scene_tasks", type=int, default=2,
                        help="v3: + candidate tasks closest in first-frame scene (text_tasks + this without instruction)")
    parser.add_argument("--p_text_drop", type=float, default=0.3,
                        help="v3: probability that a training sample's retrieval does not use its instruction")
    parser.add_argument("--hard_distractor", type=int, default=1,
                        help="v3: distractors / negatives = best-aligned other-task candidates (0: random other tasks)")
    parser.add_argument("--generic", choices=("none", "mlp"), default="none",
                        help="v4: mlp adds a reference-independent branch acting on every sample (core.GenericResidual)")
    parser.add_argument("--generic_dim", type=int, default=256, help="v4: hidden width of the generic MLPs")
    parser.add_argument("--generic_time", type=int, default=1, help="v4: condition the generic MLPs on the flow time")
    parser.add_argument("--generic_cond_rank", type=int, default=64,
                        help="v4: bottleneck of the generic MLPs' flow-time projection (0 = full rank)")
    parser.add_argument("--relevance", choices=("none", "gate"), default="none",
                        help="v6: gate = per-reference relevance gate with a null option (core.ExperienceAdapter); "
                             "train it on top of a generic branch (--generic mlp, two stages), both stages with gate")
    parser.add_argument("--train_parts", choices=("all", "generic", "reference"), default="all",
                        help="v4 stages: generic = only the generic branch, every sample without references (stage 1); "
                             "reference = only the reference branch, the generic one frozen (stage 2, --init_from the "
                             "stage-1 adapter); all = both (joint)")
    parser.add_argument("--resume", default=None,
                        help="Continue a run in --out_dir from this adapter_step*.pt: its weights, the step count (LR "
                             "schedule, remaining steps) and the best-so-far values of the run's train_log.jsonl; the "
                             "optimizer state from out_dir/train_state.pt when it is of that step, else fresh AdamW "
                             "moments with a --resume_warmup ramp. Batches and draws are new (not the lost ones)")
    parser.add_argument("--resume_warmup", type=int, default=100,
                        help="--resume without optimizer state: LR ramp over this many steps")
    parser.add_argument("--init_from", default=None,
                        help="Adapter checkpoint to start from (same configuration), e.g. the v4 stage-1 adapter")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--episodes_per_step", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2, help="(noise, time) draws per sample and step")
    parser.add_argument("--micro_batches", type=int, default=1,
                        help="Process each step's samples in this many consecutive slices (same gradient, less memory)")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pool_per_task", type=int, default=0,
                        help="v3: keep at most this many candidates per task, those with the most similar first "
                             "frames (as the runtime's --pace.pool_size; 0 = all). Large libraries (libero_plus: "
                             "~360 demos per task) need it")
    parser.add_argument("--val_frac", type=float, default=0.05)
    parser.add_argument("--val_episodes", type=int, default=32)
    parser.add_argument("--val_group", type=int, default=0,
                        help="Held-out demos per validation batch (0 = --episodes_per_step); the validation calls and "
                             "draws depend on it, so runs compared by validation need the same value (v2 / v3: 8)")
    parser.add_argument("--val_every", type=int, default=500)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--check_every", type=int, default=200, help="Compare dataset and library chunks (0 = never)")
    parser.add_argument("--max_episodes", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    assert args.num_refs <= args.max_refs
    if bool(args.rollouts) == bool(args.dataset_repo_id):
        raise SystemExit("give either --dataset_repo_id (demos) or --rollouts (v7), not both")
    if args.rollouts:
        if args.retrieval != "v3":
            raise SystemExit("--rollouts needs --retrieval v3")
        # the own variant's other attempts are near-identical scenes by construction; --p_same_variant decides
        args.same_scene_threshold = 2.0
    if args.state_weight is None:
        args.state_weight = 0.3 if args.retrieval == "v3" else 0.0
    if args.train_parts != "all" and args.generic == "none":
        raise SystemExit("--train_parts generic / reference needs --generic mlp")
    if args.train_parts == "generic":  # stage 1: a plain fine-tune of the generic branch, no references at all
        args.p_none, args.adhere = 1.0, "none"
    if args.train_parts == "reference" and not args.init_from:
        raise SystemExit("--train_parts reference trains on top of a trained generic branch: pass --init_from")

    rank, world = setup_distributed()
    torch.manual_seed(args.seed)  # identical adapter initialisation in every process (also broadcast below)
    rng = np.random.default_rng(args.seed)
    # one process: the single stream as before; several: each process draws its own batches
    draw_rng = rng if world == 1 else np.random.default_rng((args.seed, 1000 + rank))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        with open(out_dir / "args.json", "w") as f:
            json.dump({**vars(args), "world_size": world}, f, indent=2)

    policy, preprocessor = load_policy(args.policy_path, "cuda", n_action_steps=args.n_action_steps)
    store, demos, make_dataset = load_training_data(policy, args)
    dataset = make_dataset()
    prompts = PromptMap(args.prompt_override, args.prompt_map)
    # targets (`train_indices`), held-out targets, references (`universe`): see split_episodes
    train_indices, val_indices, universe = split_episodes(demos, args, rng)
    task_sims = None
    if args.retrieval == "v3":
        if not demos.has_state:
            raise SystemExit(f"--retrieval v3 aligns on robot states, which {args.library} lacks: rebuild it with "
                             "this pace.build_library")
        tasks = sorted(demos.by_task)
        task_sims = instruction_sims(policy, store, tasks)
        others = task_sims.masked_fill(torch.eye(len(tasks), dtype=torch.bool), float("nan"))
        print(f"[v3] instruction similarity between different tasks: median {others.nanmedian():.3f}, max "
              f"{others.nan_to_num(-1).max():.3f}", flush=True)
    # v9: corrective samples of the training mixed variants, the corrective test's of the held-out ones
    corr_train, corr_val, succ_val = {}, {}, {}
    if demos.rollouts and (args.p_corrective > 0 or args.corr_val):
        kind = variant_outcomes(demos)
        mixed = {v for v, k in kind.items() if k == "mixed"}
        branch = args.corr_label == "branch"
        # --corr_label branch: the aligned items (same phase) still give the success side of the corrective test
        aligned_args = argparse.Namespace(**{**vars(args), "corr_label": "same_phase"}) if branch else args
        for name, part in (("train", train_indices), ("val", val_indices)):
            variants = mixed & {demos.variant[i] for i in part}
            if name == "val" or not branch:
                items, n_pairs, n_parted = corrective_items(demos, variants, aligned_args)
                if name == "train":
                    corr_train = items if args.p_corrective > 0 else {}
                else:
                    corr_val, succ_val = items, success_side(items)
                print(f"[v9] corrective {name}: {sum(len(v) for v in items.values())} calls of {len(items)} "
                      f"failures; {n_parted} of {n_pairs} success-failure pairs part (thr {args.corr_thr}, window "
                      f"{args.corr_window})" + (" (the success side only)" if branch else ""), flush=True)
            if branch:
                items, n = branch_items(demos, variants, args)
                if name == "train":
                    corr_train = items if args.p_corrective > 0 else {}
                else:
                    corr_val = items
                print(f"[v9] branch labels {name}: {n['points']} points ({n['files']} files) -> {n['labeled']} with "
                      f"a successful branch ({n['succeeded']} of {n['branches']} branches), {len(items)} failures; "
                      f"{n['mismatched']} mismatched", flush=True)
                if name == "train" and args.p_corrective > 0 and not items:
                    raise SystemExit(f"--corr_label branch: no labeled training points in {args.branches}/"
                                     f"{args.branch_glob}")
    # v9 v2: recovery samples of the training variants, the recovery test's of the held-out ones
    rec_train, rec_val = {}, {}
    if demos.rollouts and args.recovery:
        variants = set(demos.variant)
        for name, part in (("train", variants - demos.held_variants), ("val", variants & demos.held_variants)):
            items, n = recovery_items(demos, part, args)
            print(f"[v9b] recovery {name}: {n['branches']} successful branches of {n['points']} branch points "
                  f"({n['no_success']} without, {n['with_failure']} with a rendered failed branch); {n['calls']} "
                  f"calls after the branch points, {n['window_calls']} in the window {tuple(args.rec_window)}",
                  flush=True)
            if name == "train":
                rec_train = items if args.p_recovery > 0 else {}
            else:
                rec_val = recovery_val_items(demos, items, args)
                print(f"[v9b] recovery test: {sum(len(v) for v in rec_val.values())} calls of {len(rec_val)} "
                      f"held-out branches", flush=True)
        if args.p_recovery > 0 and not rec_train:
            raise SystemExit(f"--p_recovery {args.p_recovery}: no successful rendered branch in {args.recovery}")
    queries = set(train_indices) | set(val_indices) | set(corr_train) | set(corr_val) | set(succ_val) \
        | set(rec_train) | set(rec_val)
    alignments = Alignments(demos, universe, args, task_sims, only=sorted(queries) if demos.rollouts else None)
    print(f"episodes: train={len(train_indices)} val={len(val_indices)}"
          + (f" (targets: successful attempts; references: {len(universe)} attempts of the training variants)"
             if demos.rollouts else ""), flush=True)

    if args.encoder == "v2" and not demos.features:
        raise SystemExit(f"--encoder v2 reads image tokens and robot states, which {args.library} lacks: rebuild it "
                         "with pace.build_library --features 1")
    adapter = build_adapter(policy, inject_layers=args.inject_layers, max_refs=args.max_refs,
                            input_inject=bool(args.input_inject), position=args.position,
                            time_cond=bool(args.time_cond), encoder=args.encoder,
                            feature_shapes=demos.feature_shapes(), enc_dim=args.enc_dim, enc_layers=args.enc_layers,
                            enc_heads=args.enc_heads, act_segments=args.act_segments, retrieval=args.retrieval,
                            generic=args.generic, generic_dim=args.generic_dim, generic_time=bool(args.generic_time),
                            generic_cond_rank=args.generic_cond_rank, relevance=args.relevance,
                            ref_isolation=bool(args.ref_isolation))
    if args.relevance == "gate" and args.generic == "none" and not args.rollouts:  # v7: the host fits its own rollouts
        print("[warn] --relevance gate without a generic branch: declining references then costs the domain "
              "adaptation the reference branch carries, so the gate may never learn to close", flush=True)
    start_step, resume_state = 0, None
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert ck["config"] == adapter.config, f"{args.resume}: config {ck['config']} vs {adapter.config}"
        adapter.load_state_dict(ck["state_dict"])
        start_step = int(ck["step"])
        state_path = out_dir / "train_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            resume_state = state if int(state["step"]) == start_step else None
        print(f"[resume] adapter weights from {args.resume} (step {start_step}); optimizer state "
              f"{'from ' + str(state_path) if resume_state else 'fresh (LR ramp over %d steps)' % args.resume_warmup}",
              flush=True)
        # new streams: the batches after the checkpoint are not the lost ones
        draw_rng = np.random.default_rng((args.seed, 1000 + rank, start_step))
    if args.init_from:
        start = torch.load(args.init_from, map_location="cpu", weights_only=False)
        assert start["config"] == adapter.config, f"{args.init_from}: config {start['config']} vs {adapter.config}"
        adapter.load_state_dict(start["state_dict"])
        print(f"[init] adapter weights from {args.init_from} (step {start.get('step')})", flush=True)
    if args.train_parts != "all":
        trained = adapter.generic_parameters() if args.train_parts == "generic" else adapter.reference_parameters()
        adapter.requires_grad_(False)
        for p in trained:
            p.requires_grad_(True)
    # explicit adapter inputs only: the library's keys are already computed
    pace = PI05Experience(policy, adapter, runtime_kwargs={"retrieval": args.retrieval})
    adapter.train()
    params = [p for p in adapter.parameters() if p.requires_grad]
    if world > 1:
        for p in adapter.parameters():
            dist.broadcast(p.data, 0)
        torch.manual_seed(args.seed + rank + 7919 * start_step)  # different (noise, time) draws per process
        print(f"[dist] rank {rank}/{world} on {torch.cuda.get_device_name()}", flush=True)
    print(f"adapter parameters: {sum(p.numel() for p in params) / 1e6:.1f}M trained of "
          f"{sum(p.numel() for p in adapter.parameters()) / 1e6:.1f}M (encoder {adapter.encoder}, retrieval "
          f"{args.retrieval}, generic {adapter.generic}, train_parts {args.train_parts}), expert layers {adapter.inject_layers} ({adapter.position}, time_cond={adapter.time_cond}), "
          f"config {adapter.config}", flush=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_lambda(local):
        step = start_step + local  # --resume: the schedule continues at the checkpoint's step
        if step < args.warmup:
            factor = (step + 1) / args.warmup
        else:
            progress = (step - args.warmup) / max(1, args.steps - args.warmup)
            factor = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        if start_step and resume_state is None:  # fresh moments: ramp up again
            factor *= min(1.0, (local + 1) / max(1, args.resume_warmup))
        return factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
    log = open(out_dir / "train_log.jsonl", "a") if rank == 0 else None

    def record(row):
        if rank:
            return
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()

    target_p = target_weights(demos, train_indices, args)
    corr_fails = sorted(corr_train)
    n_corr = int(round(args.p_corrective * args.episodes_per_step)) if corr_fails else 0
    rec_eps = sorted(rec_train)
    n_rec = int(round(args.p_recovery * args.episodes_per_step)) if rec_eps else 0
    rec_p = np.array([rec_train[e][3] for e in rec_eps]) if rec_eps else None
    if rec_eps:
        rec_p = rec_p / rec_p.sum()
        print(f"[v9b] {n_rec} recovery samples of {args.episodes_per_step} episodes per step and process "
              f"(contrast weight {args.rec_contrast}, margin {args.rec_margin})", flush=True)

    def draw():
        n_plain = args.episodes_per_step - n_corr - n_rec
        if target_p is None:
            picks = draw_rng.integers(len(train_indices), size=n_plain)
        else:  # v7 --mixed_weight
            picks = draw_rng.choice(len(train_indices), size=n_plain, p=target_p)
        episodes = [train_indices[k] for k in picks]
        frames = [sample_frames(demos.actions[e].shape[0], args.group_size, draw_rng) for e in episodes]
        targets = [None] * len(episodes)
        for _ in range(n_corr):  # v9: a failure's corrective calls, the aligned success's actions as targets
            f = corr_fails[int(draw_rng.integers(len(corr_fails)))]
            its = corr_train[f]
            chosen = [its[k] for k in draw_rng.choice(len(its), size=args.group_size,
                                                      replace=len(its) < args.group_size)]
            episodes.append(f)
            frames.append(np.array([j for j, _, _ in chosen]))
            targets.append(torch.stack([item_label(demos, s, t).to(demos.actions[f].dtype) for _, s, t in chosen]))
        alts = [False] * len(episodes)
        for _ in range(n_rec):  # v9 v2: a successful branch's calls after its branch point, its own chunks as targets
            e = rec_eps[int(draw_rng.choice(len(rec_eps), p=rec_p))]
            calls, w, alt, _ = rec_train[e]
            k = draw_rng.choice(len(calls), size=args.group_size, replace=len(calls) < args.group_size, p=w / w.sum())
            frame = np.sort(calls[k])
            episodes.append(e)
            frames.append(frame)
            targets.append(None)
            alts.append(recovery_contrast(demos, alt, frame))
        conditions = [alignments.training_condition(draw_rng) for _ in episodes]
        refs = [alignments.references(e, "real" if c == "unaligned" else c, draw_rng)
                for e, c in zip(episodes, conditions)]
        negs = []
        for e, c, ref in zip(episodes, conditions, refs):
            if args.adhere == "none" or c != args.ref_content or ref is None:
                negs.append(None)
                continue
            other = alignments.references(e, "other_task", draw_rng) \
                if alignments.has_other_tasks and draw_rng.random() < args.p_neg_other else None
            negs.append((other, False) if other is not None else (ref, True))
        return make_plan(demos, episodes, refs, conditions, frames, draw_rng, args, negs, targets, alts)

    sampler = PlanSampler(draw, args.steps - start_step)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, collate_fn=default_collate,
                        persistent_workers=False)
    # Fork the loader's workers BEFORE this process decodes any video: lerobot keeps open torchcodec decoders in a
    # module-level cache, and workers forked after the validation frames were decoded inherit them and fail with
    # "Could not push packet to decoder: Invalid data" (a separate dataset object does not help: the cache is global)
    batches = iter(loader)
    val, val_batches = {}, None
    mixed_val = None
    if demos.rollouts:  # v7: the held-out successes of mixed variants, reported apart (val10_mixed_*)
        kind = variant_outcomes(demos)
        mixed_val = {i for i in val_indices if kind[demos.variant[i]] == "mixed"}
    corr_batches = rec_batches = None
    if rank == 0:  # validation, logging and saving are rank 0's; the others wait in the next gradient all-reduce
        val_dataset = make_dataset()
        val_batches = build_validation(demos, alignments, val_indices, val_dataset, args)
        if corr_val:  # failure side (the correction) and success side (the successes must keep their way)
            corr_batches = (build_corrective_validation(demos, alignments, corr_val, val_dataset, args),
                            build_corrective_validation(demos, alignments, succ_val, val_dataset, args))
        if rec_val:  # v9 v2: the recovery test
            rec_batches = build_corrective_validation(demos, alignments, rec_val, val_dataset, args)
        val = validate(policy, pace, val_batches, preprocessor, prompts, args, mixed=mixed_val)
        if corr_batches:
            val.update(validate_both_sides(policy, pace, corr_batches, preprocessor, prompts, args))
        if rec_batches:
            val.update(validate_recovery(policy, pace, rec_batches, preprocessor, prompts, args))
        if not start_step:
            record({"step": 0, **val})  # zero-initialized adapter: every condition equals `none` here
        else:  # must reproduce the checkpoint's own validation
            print("[resume] validation at the checkpoint:",
                  {k: round(val[k], 5) for k in ("val10_real", "val10_none", "val_rec_real_gain") if k in val}, flush=True)
    best_corr = best_rec = float("-inf")
    running, started = collections.defaultdict(float), time.time()
    relevance_running, relevance_steps = collections.defaultdict(float), 0
    best_val = val.get("val_none", float("inf")) if args.train_parts == "generic" else float("inf")
    if start_step and (out_dir / "train_log.jsonl").exists():  # --resume: the run's best values up to its step
        for line in open(out_dir / "train_log.jsonl"):
            row = json.loads(line)
            if row.get("step", 0) > start_step:
                continue
            best_corr = max(best_corr, row.get("best_val_corr_real_gain", float("-inf")))
            best_rec = max(best_rec, row.get("best_val_rec_real_gain", float("-inf")))
            best_val = min(best_val, row.get("best_val10_real", row.get("best_val_none", float("inf"))))
        print(f"[resume] best so far: val10_real {best_val}, rec gain {best_rec}, corr gain {best_corr}", flush=True)
    for step, raw in enumerate(batches, start_step + 1):
        plan = sampler.plans.popleft()
        batch = prepare(raw, preprocessor, prompts, plan["chunks"])
        if args.check_every and (step == start_step + 1 or step % args.check_every == 0):
            check_chunks(batch, plan, step)
        optimizer.zero_grad(set_to_none=True)
        loss, backward, stats, relevance = training_loss(policy, pace, batch, plan, args)
        if world > 1:
            flag = torch.tensor(float(backward), device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            backward = bool(flag)
            if backward:
                average_gradients(params, world)
        if backward:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
            optimizer.step()
        else:
            grad_norm = 0.0
            running["n_skipped"] += 1
        scheduler.step()

        running["loss"] += loss
        for name, value in stats.items():
            running[name] += value
        if relevance:  # averaged over the steps that had references
            for name, value in relevance.items():
                relevance_running[name] += value
            relevance_steps += 1
        for condition in plan["conditions"]:
            running[f"n_{condition}"] += 1
        if step % args.log_every == 0:
            gates = adapter.gates()
            averaged = {k: running.pop(k) / args.log_every for k in list(running) if not k.startswith("n_")}
            record({
                "step": step, **averaged, "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm, "steps_per_s": (step - start_step) / (time.time() - started),
                "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
                "gate_mean": float(np.mean(list(gates.values()))), "gate_max": float(np.max(np.abs(list(gates.values())))),
                **({"generic_gate_mean": float(np.mean(list(adapter.generic_gates().values())))}
                   if adapter.has_generic else {}),
                # v6: g and the injection's weight on the references, mean over the logged steps' own references
                **{name: value / relevance_steps for name, value in relevance_running.items()},
                **running,
            })
            running.clear()
            relevance_running.clear()
            relevance_steps = 0
        if rank == 0 and (step % args.val_every == 0 or step == args.steps):
            val = validate(policy, pace, val_batches, preprocessor, prompts, args, mixed=mixed_val)
            if corr_batches:
                val.update(validate_both_sides(policy, pace, corr_batches, preprocessor, prompts, args))
            if rec_batches:
                val.update(validate_recovery(policy, pace, rec_batches, preprocessor, prompts, args))
            record({"step": step, **val, **adapter.gates(), **adapter.generic_gates()})
            if rec_batches and val["val_rec_real_gain"] > best_rec:  # v9 v2: the largest recovery-test gain
                best_rec = val["val_rec_real_gain"]
                save_adapter(adapter, out_dir / "adapter_best_rec.pt", step=step, args=vars(args), val=val,
                             policy_path=args.policy_path, key_info=store.info, vmax=args.vmax, gamma=args.gamma,
                             align=args.align, retrieval=args.retrieval, generic=args.generic,
                             train_parts=args.train_parts)
                record({"step": step, "best_val_rec_real_gain": best_rec, "saved": "adapter_best_rec.pt"})
            if corr_batches and val["val_corr_real_gain"] > best_corr:  # v9: the largest corrective gain
                best_corr = val["val_corr_real_gain"]
                save_adapter(adapter, out_dir / "adapter_best_corr.pt", step=step, args=vars(args), val=val,
                             policy_path=args.policy_path, key_info=store.info, vmax=args.vmax, gamma=args.gamma,
                             align=args.align, retrieval=args.retrieval, generic=args.generic,
                             train_parts=args.train_parts)
                record({"step": step, "best_val_corr_real_gain": best_corr, "saved": "adapter_best_corr.pt"})
            # keep the adapter with the lowest held-out loss (it may overfit later): stage 1 of v4 (a plain fine-tune)
            # by val_none; v7 (a few thousand rollouts) by val10_real (the executed first steps of a prediction)
            best_key = "val_none" if args.train_parts == "generic" else ("val10_real" if demos.rollouts else None)
            if best_key and val.get(best_key, float("inf")) < best_val:
                best_val = val[best_key]
                save_adapter(adapter, out_dir / "adapter_best.pt", step=step, args=vars(args), val=val,
                             policy_path=args.policy_path, key_info=store.info, vmax=args.vmax, gamma=args.gamma,
                             align=args.align, retrieval=args.retrieval, generic=args.generic,
                             train_parts=args.train_parts)
                record({"step": step, f"best_{best_key}": best_val, "saved": "adapter_best.pt"})
        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            extra = {"step": step, "args": vars(args), "val": val, "policy_path": args.policy_path,
                     "key_info": store.info, "vmax": args.vmax, "gamma": args.gamma, "align": args.align,
                     "retrieval": args.retrieval, "generic": args.generic, "train_parts": args.train_parts}
            save_adapter(adapter, out_dir / f"adapter_step{step:06d}.pt", **extra)
            save_adapter(adapter, out_dir / "adapter_last.pt", **extra)
            # --resume: the optimizer state of the latest save (written beside, then renamed)
            torch.save({"step": step, "optimizer": optimizer.state_dict()}, out_dir / "train_state.pt.tmp")
            os.replace(out_dir / "train_state.pt.tmp", out_dir / "train_state.pt")
    if log:
        log.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
