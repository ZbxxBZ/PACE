"""
Evaluate pi0.5 (optionally with PACE) on a chosen list of LIBERO / LIBERO-plus task variants, batching DIFFERENT
variants into one vector env (lerobot-eval batches only episodes of one task, and its parallel-task mode shares one
policy across threads without a lock; LIBERO-plus evaluates every variant once, so lerobot-eval would run one env at a
time). Each variant runs one episode from its own fixed init state (episode_index 0) with seed --seed + its position
in the list; the policy, processors, env construction (lerobot.envs.libero._make_env_fns, async workers that freeze
after their episode) and the rollout loop are lerobot-eval's.

LIBERO-plus instruction fix (upstream issue sylvestf/LIBERO-plus#64, open at 4976dc3): outside the language category
`task.language` is built from the file name, so the policy would be told e.g. "... view 0 0 100 0 0 initstate 236".
With --fix_language (default for libero_plus) the policy is told the base task's training instruction instead (the
dataset's task string, matched from the variant's file name), and in the language category the paraphrase itself.
The classic retrieval looks the base instruction up (so the language category is not a fair test of it); the v3
retrieval (--pace.retrieval v3) compares the instruction the policy is told with the stored tasks'
(--pace.retrieval_text shown, default).

Variant selection: --tasks_file (JSON list of task ids, e.g. the sample.json a previous run wrote) or, for
libero_plus, --per_category N variants of each perturbation category (LIBERO-plus task_classification.json) drawn
with --sample_seed; for libero, every task of the suite x --episodes.

Rounds and accumulation (cross-episode experience): --rounds R runs the whole list R times, every round with the same
init states and seeds (so rounds and runs pair episode by episode; rows carry "round"). With --pace.accumulate every
batch's rollouts join the store as soon as the batch ends (later batches of the round and every later round can
retrieve them; the retrieval reads successes only unless --pace.include_failures) and are saved to
<out_dir>/rollouts/rollouts_r<round>_b<batch>.pt; a resumed run reloads them, in order, after the library. Without PACE
--rounds R simply re-runs the base policy R times (its run-to-run noise). Compare runs with pace.compare_runs.

v7 training data (--record_frames, see pace/rollouts.py): with --pace.condition none --pace.accumulate and an adapter
with the v2 encoder the policy runs as the base policy, every rollout is stored with its image tokens and executed
chunks, and every policy call's observation goes to <out_dir>/rollouts/frames_<tag>.pt; the rounds are then repeated
attempts (the policy's noise draws differ between rounds).

    python -m pace.eval_suite --env_type libero_plus --suite libero_10 --per_category 40 --out_dir eval/plus_l10_base
    python -m pace.eval_suite ... --pace.condition real --pace.library libs/l10.pt --pace.adapter runs/x/adapter_last.pt
"""

import argparse
import collections
import functools
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from .common import PromptMap, load_policy
from .core import normalize_instruction
from .eval import FixedTasks, PaceRollout, add_pace_args, attach_pace, rollout_files
from .rollouts import FrameRecorder

SCENE_PREFIX = re.compile(r"^(?:[A-Z]+_)+SCENE\d+_")


def base_instructions(dataset_root: str) -> list:
    """Training instructions of the LIBERO dataset (lerobot/libero meta/tasks.parquet)."""
    import pandas as pd

    tasks = pd.read_parquet(Path(dataset_root) / "meta" / "tasks.parquet")
    return list(tasks.index)


def match_base(variant_name: str, bases: list) -> str:
    """The base instruction a variant's file name starts with (longest match)."""
    stem = SCENE_PREFIX.sub("", variant_name)
    hits = [b for b in bases if stem == b.replace(" ", "_") or stem.startswith(b.replace(" ", "_") + "_")]
    assert hits, f"no base instruction for {variant_name}"
    return max(hits, key=len)


LANGUAGE_CATEGORY = "Language Instructions"


def classification(suite_name: str) -> dict:
    """
    LIBERO-plus variant name -> (category, difficulty). The file names some suites' variants with their scene prefix
    (libero_10) and others without (libero_spatial); both spellings are keys.
    """
    import libero.libero

    path = Path(libero.libero.__file__).parent / "benchmark" / "task_classification.json"
    data = json.load(open(path))
    out = {}
    for e in data[suite_name]:
        value = (e["category"], e.get("difficulty_level"))
        out[e["name"]] = value
        out[SCENE_PREFIX.sub("", e["name"])] = value
    return out


def variant_class(classes: dict, name: str):
    if not classes:
        return "Original", None
    hit = classes.get(name) or classes.get(SCENE_PREFIX.sub("", name))
    assert hit is not None, f"{name} is not in task_classification.json"
    return hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", choices=("libero", "libero_plus"), required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--policy_path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--n_action_steps", type=int, default=10)
    parser.add_argument("--dataset_root", default="/root/pace/cache/hf/lerobot/lerobot/libero",
                        help="LIBERO dataset (its tasks.parquet gives the training instructions)")
    parser.add_argument("--tasks_file", default=None, help="JSON list of task ids to evaluate")
    parser.add_argument("--per_category", type=int, default=40, help="libero_plus sample size per category")
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1, help="Episodes per task (libero: init states 0..n-1)")
    parser.add_argument("--fix_language", type=int, default=None, help="Default: 1 for libero_plus, 0 for libero")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max_batches", type=int, default=0, help="0 = all (debugging aid)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Run the list this many times (same init states and seeds every round); with "
                             "--pace.accumulate the store keeps growing across batches and rounds")
    parser.add_argument("--record_frames", action="store_true",
                        help="v7 training data: also record every policy call's observation next to the rollouts "
                             "(pace.rollouts; needs --pace.accumulate and an adapter with the v2 encoder, run with "
                             "--pace.condition none so the policy runs as the base policy)")
    add_pace_args(parser, condition_default=None)
    args = parser.parse_args()
    use_pace = args.condition is not None
    if args.record_frames and not (use_pace and args.accumulate):
        raise SystemExit("--record_frames needs --pace.accumulate (the frames are stored next to the rollouts)")
    fix_language = bool(args.env_type == "libero_plus" if args.fix_language is None else args.fix_language)

    from lerobot.envs.configs import LiberoEnv, LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _get_suite, _LazyAsyncVectorEnv, _make_env_fns, parse_camera_names
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout as lerobot_rollout
    from lerobot.utils.random_utils import set_seed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_cfg = (LiberoPlusEnv if args.env_type == "libero_plus" else LiberoEnv)(task=args.suite)
    suite = _get_suite(args.suite)

    # --- variants ---
    bases = base_instructions(args.dataset_root)
    classes = classification(args.suite) if args.env_type == "libero_plus" else {}
    if args.tasks_file:
        task_ids = [int(t) for t in json.load(open(args.tasks_file))]
    elif args.env_type == "libero_plus":
        by_cat = collections.defaultdict(list)
        for tid in range(suite.n_tasks):
            by_cat[variant_class(classes, suite.get_task(tid).name)[0]].append(tid)
        rng = np.random.default_rng(args.sample_seed)
        task_ids = sorted(t for cat in sorted(by_cat) for t in
                          rng.choice(by_cat[cat], min(args.per_category, len(by_cat[cat])), replace=False).tolist())
        print("[suite] categories:", {c: len(v) for c, v in sorted(by_cat.items())}, flush=True)
    else:
        task_ids = list(range(suite.n_tasks))
    with open(out_dir / "sample.json", "w") as f:
        json.dump(task_ids, f)
    entries = []
    for tid in task_ids:
        task = suite.get_task(tid)
        cat, diff = variant_class(classes, task.name)
        base = match_base(task.name, bases)
        shown = task.language
        if fix_language:
            shown = task.language if cat == LANGUAGE_CATEGORY else base
        for ep in range(args.episodes):
            entries.append({"task_id": tid, "name": task.name, "category": cat, "difficulty": diff, "base": base,
                            "shown": shown, "raw_language": task.language, "init_index": ep})
    print(f"[suite] {args.env_type}/{args.suite}: {len(task_ids)} variants x {args.episodes} = {len(entries)} "
          f"episodes, batch {args.batch_size}, fix_language={fix_language}", flush=True)
    print("[suite] e.g.", [(e["category"], e["raw_language"][:70], "->", e["shown"][:70]) for e in entries[:3]],
          flush=True)

    # --- policy and processors, as lerobot-eval builds them ---
    set_seed(args.seed)
    policy, _ = load_policy(args.policy_path, "cuda", n_action_steps=args.n_action_steps)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": {}}},
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    rollout_dir = out_dir / "rollouts"
    if use_pace and args.accumulate:  # resume: the rollouts an earlier (interrupted) run stored, after the library
        previous = rollout_files(rollout_dir)
        if previous:
            args.library = list(args.library) + [str(p) for p in previous]
            print(f"[suite] resuming the store: {len(previous)} rollout files from {rollout_dir}", flush=True)
    log = open(out_dir / "episodes.jsonl", "a")
    if use_pace:
        pace = attach_pace(policy, args)
        if args.record_frames and not pace.features:
            raise SystemExit("--record_frames: the rollouts need image tokens (training data for the v2 encoder); "
                             "pass an adapter with the v2 encoder or --pace.record_features (--pace.condition none keeps "
                             "the base policy)")
        runner = PaceRollout(lerobot_rollout, lambda: pace, args, PromptMap(args.prompt_override, args.prompt_map), log,
                             rollout_dir if args.accumulate else None)

    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    camera_names = parse_camera_names(env_cfg.camera_name)
    started = time.time()
    n_done = n_success = 0
    done_keys = set()
    if (out_dir / "episodes.jsonl").exists():  # resume: skip the episodes an earlier (interrupted) run recorded
        done_keys = {(r.get("round", 1), r["task_id"], r.get("init_index", 0))
                     for r in map(json.loads, open(out_dir / "episodes.jsonl"))}
        print(f"[suite] resuming: {len(done_keys)} episodes already recorded", flush=True)
    for position, e in enumerate(entries):  # the seed depends only on the position in the list (same every round)
        e["seed"] = args.seed + position
    # batches never mix rounds; their numbering (rollout file names) is that of an uninterrupted run
    batches = []
    for rnd in range(1, args.rounds + 1):
        listed = [dict(e, round=rnd) for e in entries]
        for i in range(0, len(listed), args.batch_size):
            batch = [e for e in listed[i : i + args.batch_size]
                     if (rnd, e["task_id"], e["init_index"]) not in done_keys]
            if batch:
                batches.append((rnd, i // args.batch_size, batch))
    for b, (rnd, number, batch) in enumerate(batches):
        if args.max_batches and b >= args.max_batches:
            break
        fns = []
        for e in batch:
            fn = _make_env_fns(suite=suite, suite_name=args.suite, task_id=e["task_id"], n_envs=1,
                               camera_names=camera_names, episode_length=env_cfg.episode_length,
                               init_states=env_cfg.init_states, gym_kwargs=gym_kwargs, control_mode=env_cfg.control_mode,
                               camera_name_mapping=env_cfg.camera_name_mapping,
                               is_libero_plus=env_cfg.is_libero_plus)[0]
            if e["init_index"]:  # libero: episode k of a task starts from its init state k (episode_index)
                fn = functools.partial(fn.func, e["init_index"], **fn.keywords)
            fns.append(fn)
        env = _LazyAsyncVectorEnv(fns)
        seeds = [e["seed"] for e in batch]
        kw = dict(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                  preprocessor=preprocessor, postprocessor=postprocessor, seeds=seeds)
        try:
            if use_pace:
                recorder = None
                if args.record_frames:  # every policy call's observation, as the policy preprocessor receives it
                    recorder = FrameRecorder(kw["env_preprocessor"], policy.config.n_action_steps)
                    kw["env_preprocessor"] = recorder
                ret = runner(tasks=[e["base"] for e in batch], shown=[e["shown"] for e in batch],
                             row_extra=[{k: e[k] for k in ("round", "task_id", "name", "category", "difficulty",
                                                           "init_index", "base", "shown")}
                                        for e in batch],
                             names=[f"rollout/{args.condition}/r{e['round']:02d}/t{e['task_id']:05d}/"
                                    f"i{e['init_index']:03d}" for e in batch],
                             runtime_seeds=[(args.seed * 1000003 + 7919 * e["task_id"] + 104729 * e["init_index"]
                                             + 15485863 * e["round"]) % (2 ** 31) for e in batch],
                             save_tag=f"r{rnd:02d}_b{number:04d}", recorder=recorder, **kw)
            else:
                kw["env_preprocessor"] = FixedTasks(env_pre, [e["shown"] for e in batch])
                ret = lerobot_rollout(**kw)
                done = ret["done"].bool()
                success = ret["success"].bool().any(dim=1).tolist()
                T = done.shape[1]
                for i, e in enumerate(batch):
                    last = int(torch.argmax(done[i].int())) if bool(done[i].any()) else T - 1
                    log.write(json.dumps({**e, "success": bool(success[i]),
                                          "env_steps": last + 1}) + "\n")
                log.flush()
        finally:
            env.close()
        rows = [json.loads(l) for l in open(out_dir / "episodes.jsonl")]
        this_round = [r for r in rows if r.get("round", 1) == rnd]
        store = f", store {len(pace.store)} episodes" if use_pace and pace.store is not None else ""
        print(f"[suite] batch {b + 1}/{len(batches)}: round {rnd} {len(this_round)} episodes, success "
              f"{100 * np.mean([r['success'] for r in this_round]):.1f}%{store} ({time.time() - started:.0f}s)",
              flush=True)
    log.close()

    rows = [json.loads(l) for l in open(out_dir / "episodes.jsonl")]

    def summarize(subset):
        per_cat = collections.defaultdict(list)
        for r in subset:
            per_cat[r["category"]].append(r["success"])
        out = {"overall": 100 * float(np.mean([r["success"] for r in subset])), "n": len(subset),
               "per_category": {c: {"success": 100 * float(np.mean(v)), "n": len(v)} for c, v in sorted(per_cat.items())}}
        rollout_frac = [r["top_ref_rollout_frac"] for r in subset if "top_ref_rollout_frac" in r]
        if rollout_frac:  # how often the top reference was a rollout (accumulated experience) rather than a demo
            out["top_ref_rollout_frac"] = float(np.mean(rollout_frac))
        return out

    rounds = sorted({r.get("round", 1) for r in rows})
    summary = {**summarize(rows), "rounds": {str(k): summarize([r for r in rows if r.get("round", 1) == k])
                                             for k in rounds},
               "args": vars(args)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("overall", "n", "per_category", "rounds")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
