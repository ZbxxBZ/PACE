"""
`lerobot-eval` with PACE attached to the pi0.5 policy. Every `--pace.*` option is read here; everything else goes to
lerobot-eval unchanged (env, policy, seeds, videos, output dir). Two lerobot-eval functions are wrapped:
  - `make_policy`: the created policy gets a PI05Experience (adapter + experience store);
  - `rollout`: before a batch of episodes, one runtime per env is started with the env's true instruction as task
    key (the policy may be told something else: --pace.prompt_override / --pace.prompt_map); after it, the episodes
    are closed, one json line per episode goes to <out_dir>/pace_episodes.jsonl and, with --pace.accumulate, the
    rollouts join the store, so later episodes can retrieve earlier ones (cross-episode experience).
    With --pace.retrieval v3 the retrieval compares the instruction the policy is told (--pace.retrieval_text shown,
    default; `true` = the env's instruction, an oracle; `none` = no instruction) with the stored tasks', instead of
    looking the true instruction up, and the rollouts are stored under what the policy was told.
Rollouts are serialized (a lock), since the episode state lives on the shared policy.

Episodes are paired across runs by (task, episode): LIBERO env i of rollout batch r starts from init state
r * batch_size + i and lerobot-eval seeds it with seed + r * batch_size + i, so keep --eval.batch_size and --seed
equal across the conditions being compared.

Without --pace.adapter a zero-initialized adapter is attached: its output is exactly the base policy's (a check of
the attachment itself; compare with plain lerobot-eval).

    python -m pace.eval --pace.adapter runs/goal_main/adapter_last.pt --pace.library libs/libero_goal_demo.pt \
        --pace.condition real --pace.out_dir eval/goal_real -- \
        --env.type=libero --env.task=libero_goal --eval.batch_size=10 --eval.n_episodes=50 \
        --policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10 --policy.compile_model=false \
        --output_dir=eval/goal_real/lerobot

The PACE pieces (`add_pace_args`, `attach_pace`, `PaceRollout`) are shared with pace/eval_suite.py.
"""

import argparse
import json
import math
import os
import sys
import threading
import time
import zlib
from pathlib import Path

import torch

from .common import PromptMap, check_loaded
from .core import CONDITIONS, ExperienceStore, load_adapter, normalize_instruction
from .pi05 import PI05Experience, build_adapter
from .rollouts import exclusive_writes, fsync_path, save_frames


def add_pace_args(parser: argparse.ArgumentParser, condition_default="real") -> None:
    parser.add_argument("--pace.adapter", dest="adapter", default=None)
    parser.add_argument("--pace.library", dest="library", nargs="*", default=[], help="Store files (demo libraries)")
    parser.add_argument("--pace.condition", dest="condition", choices=CONDITIONS, default=condition_default)
    parser.add_argument("--pace.num_refs", dest="num_refs", type=int, default=4)
    parser.add_argument("--pace.pool_size", dest="pool_size", type=int, default=64)
    parser.add_argument("--pace.select", dest="select", choices=("similar", "recent", "random"), default="similar")
    parser.add_argument("--pace.sources", dest="sources", nargs="+", default=["demo", "rollout"])
    parser.add_argument("--pace.include_failures", dest="include_failures", action="store_true",
                        help="Failed episodes may be references too (their success flag is an adapter input); on by "
                             "default for adapters trained so (train.py --ref_failures 1)")
    parser.add_argument("--pace.vmax", dest="vmax", type=int, default=None, help="Default: the adapter's training value")
    parser.add_argument("--pace.gamma", dest="gamma", type=float, default=None)
    parser.add_argument("--pace.align", dest="align", choices=("pma", "nn"), default=None,
                        help="PMA over the history, or frame nearest neighbour (default: the adapter's training value)")
    parser.add_argument("--pace.cameras", dest="cameras", type=int, nargs="+", default=None,
                        help="Default: the store's (0 1 without a store)")
    parser.add_argument("--pace.accumulate", dest="accumulate", action="store_true",
                        help="Add every finished episode to the store (retrieval still reads successes only unless "
                             "--pace.include_failures); each batch's rollouts are saved to <out_dir>/rollouts/")
    parser.add_argument("--pace.record_features", dest="record_features", action="store_true",
                        help="Without --pace.adapter: attach a zero-initialized adapter with the v2 encoder (v3 "
                             "retrieval), so rollouts keep their image tokens and states (v7 data collection without a "
                             "trained adapter; with --pace.condition none the policy runs as the base policy)")
    parser.add_argument("--pace.feature_shapes", dest="feature_shapes", type=int, nargs=3, default=[32, 2048, 8],
                        metavar=("V", "DV", "DS"),
                        help="--pace.record_features: image tokens per call (16 per camera), their width (the "
                             "host's), state size; default LIBERO's two cameras with pi0.5")
    parser.add_argument("--pace.write_lock", dest="write_lock", default=None,
                        help="Lock file shared by runs writing to the same disk: each batch's rollout (and frame) files "
                             "are written and flushed to the disk under it, one run at a time (default: no lock)")
    parser.add_argument("--pace.write_gap", dest="write_gap", type=float, default=0.0,
                        help="With --pace.write_lock: a run writes only once this many seconds have passed since the "
                             "previous run's write ended (the disk is quiet between two writes)")
    parser.add_argument("--pace.prompt_override", dest="prompt_override", default=None)
    parser.add_argument("--pace.prompt_map", dest="prompt_map", default=None)
    parser.add_argument("--pace.seed", dest="seed", type=int, default=0, help="Runtime RNG (control conditions)")
    parser.add_argument("--pace.retrieval", dest="retrieval", choices=("classic", "v3"), default=None,
                        help="Default: the adapter's training value (classic without an adapter)")
    parser.add_argument("--pace.retrieval_text", dest="retrieval_text", choices=("shown", "true", "none"),
                        default="shown",
                        help="v3: the instruction compared with the stored tasks: the one the policy is told (fair), "
                             "the env's true one (oracle), or none")
    for name in ("state_weight", "state_sigma", "text_weight"):
        parser.add_argument(f"--pace.{name}", dest=name, type=float, default=None,
                            help="v3 (default: the adapter's training value)")
    parser.add_argument("--pace.generic", dest="generic", choices=("on", "off"), default="on",
                        help="v4 adapters: the reference-independent generic branch on (default; with --pace.condition "
                             "none this is the generic-only control) or off (the host alone when there are no references)")
    for name in ("text_tasks", "scene_tasks"):
        parser.add_argument(f"--pace.{name}", dest=name, type=int, default=None,
                            help="v3 (default: the adapter's training value)")
    parser.add_argument("--pace.max_image_cost", dest="max_image_cost", type=float, default=None,
                        help="v6 hard gate: skip a reference whose image cost (1 - key similarity) at its matched step "
                             "exceeds this (e.g. the training top-reference q90, which train.py logs as [v3] image "
                             "cost); a call left without references runs the host (plus a generic branch)")


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m pace.eval", add_help=True)
    parser.add_argument("--pace.out_dir", dest="out_dir", required=True)
    add_pace_args(parser)
    if "--" in argv:
        split = argv.index("--")
        return parser.parse_args(argv[:split]), argv[split + 1 :]
    return parser.parse_known_args(argv)


def attach_pace(policy, args) -> PI05Experience:
    """PI05Experience on `policy` from the --pace.* options (zero-initialized adapter without --pace.adapter)."""
    store = ExperienceStore.load(args.library) if args.library else None
    # Building the adapter draws random initial weights; keep it off the global RNG the evaluation seeded, so the
    # policy's noise draws (and hence episodes) stay comparable with a run without PACE
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        if args.adapter:
            adapter, ckpt = load_adapter(args.adapter)
            trained = ckpt.get("args", {})
        elif getattr(args, "record_features", False):  # zero-initialized v2 adapter: records features, adds nothing
            V, Dv, Ds = args.feature_shapes
            adapter = build_adapter(policy, encoder="v2", feature_shapes=((V, Dv), Ds), retrieval=args.retrieval or "v3")
            trained = {"retrieval": args.retrieval or "v3"}
        else:
            adapter = build_adapter(policy, retrieval=args.retrieval or "classic")
            trained = {"retrieval": args.retrieval or "classic"}
    trained_value = lambda name, default: getattr(args, name) if getattr(args, name) is not None \
        else trained.get(name, default)  # noqa: E731
    retrieval = args.retrieval or trained.get("retrieval", "classic")
    # v7 adapters trained with failed attempts among their references (--ref_failures 1) read them by default too
    include_failures = bool(args.include_failures or trained.get("ref_failures", 0))
    runtime_kwargs = dict(
        num_refs=args.num_refs, pool_size=args.pool_size, select=args.select, sources=tuple(args.sources),
        success_only=not include_failures,
        vmax=args.vmax if args.vmax is not None else int(trained.get("vmax", 3)),
        gamma=args.gamma if args.gamma is not None else float(trained.get("gamma", 0.01)),
        align=args.align or trained.get("align", "pma"),
        retrieval=retrieval,
    )
    if args.max_image_cost is not None:
        runtime_kwargs["max_image_cost"] = float(args.max_image_cost)
    if retrieval == "v3":
        state_weight = trained_value("state_weight", None)
        runtime_kwargs.update(
            state_weight=float(0.3 if state_weight is None else state_weight),
            state_sigma=float(trained_value("state_sigma", 0.3)), text_weight=float(trained_value("text_weight", 0.2)),
            text_tasks=int(trained_value("text_tasks", 3)), scene_tasks=int(trained_value("scene_tasks", 2)),
        )
    # Keys (cameras, raw key, fitted transform) follow the store; --pace.cameras must agree with it
    pace = PI05Experience(policy, adapter, store, cameras=args.cameras, runtime_kwargs=runtime_kwargs)
    pace.generic_on = pace.adapter.has_generic and args.generic == "on"
    pace.adapter.eval()
    print(f"[pace] condition={args.condition} adapter={args.adapter or 'zero-init'} generic="
          f"{'on' if pace.generic_on else ('off' if pace.adapter.has_generic else 'none')} "
          f"store={0 if store is None else len(store)} episodes runtime={runtime_kwargs}", flush=True)
    return pace


class FixedTasks:
    """The env preprocessor, with the instruction each env's policy sees fixed (the env keeps its own task)."""

    def __init__(self, inner, shown):
        self.inner, self.shown = inner, list(shown)

    def __call__(self, observation):
        observation = dict(observation)
        assert len(observation["task"]) == len(self.shown), "one shown instruction per env"
        observation["task"] = list(self.shown)
        return self.inner(observation)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def rollout_files(rollout_dir) -> list:
    """The rollout files a run saved (`PaceRollout`), in the order they were written."""
    rollout_dir = Path(rollout_dir)
    return sorted(rollout_dir.glob("rollouts_*.pt")) if rollout_dir.is_dir() else []


class PaceRollout:
    """
    Wraps lerobot-eval's `rollout`: PACE episodes begin before and end after each batch, one json line per episode.
    `tasks` (true instructions = task keys) and `shown` (what the policy is told) default to the envs'
    task_description and the prompt map; pace/eval_suite.py passes both explicitly, and may also fix the rollouts'
    store names (`names`) and the runtimes' seeds (`runtime_seeds`), so a resumed run names and seeds its episodes as
    an uninterrupted one would.

    With --pace.accumulate the batch's episodes (successes and failures, each with its outcome) join the store, and
    with a `rollout_dir` they are also written to <rollout_dir>/rollouts_<tag>.pt (only that batch's episodes, without
    the key transform, which the library holds) BEFORE the batch's json rows: the library plus these files, in name
    order, rebuild the store of any point of the run (`rollout_files`).
    """

    def __init__(self, original_rollout, pace_getter, args, prompts: PromptMap, log, rollout_dir=None):
        self.original, self.pace_getter, self.args = original_rollout, pace_getter, args
        self.prompts, self.log = prompts, log
        self.rollout_dir = Path(rollout_dir) if rollout_dir is not None else None
        self.episodes = {}
        self.batches = 0
        self.last_rows = []  # the json rows of the latest batch
        self.lock = threading.Lock()
        self.sync = bool(getattr(args, "write_lock", None))  # flush the saved files inside the shared write lock

    def save_rollouts(self, pace, start: int, tag: str):
        """
        The store's episodes from index `start` on (this batch's) -> <rollout_dir>/rollouts_<tag>.pt (with
        --pace.write_lock: flushed to the disk before this returns).
        """
        if self.rollout_dir is None or pace.store is None or len(pace.store) <= start:
            return None
        self.rollout_dir.mkdir(parents=True, exist_ok=True)
        path, k = self.rollout_dir / f"rollouts_{tag}.pt", 0
        while path.exists():  # a resumed run re-running part of a batch: keep what the first attempt stored
            k += 1
            path = self.rollout_dir / f"rollouts_{tag}_{k}.pt"
        tmp = path.with_suffix(".tmp")
        ExperienceStore(pace.store.episodes[start:], pace.store.info).save(tmp)
        if self.sync:
            fsync_path(tmp)
        os.replace(tmp, path)  # a complete file or none
        if self.sync:
            fsync_path(path.parent)
        return path

    def save_frames(self, pace, start: int, infos, metas, recorder, saved) -> None:
        """
        v7 recording: the calls `recorder` (`rollouts.FrameRecorder`) kept for this batch's stored episodes ->
        frames_<tag>.pt next to their rollout file, written before the batch's json rows like the rollouts.
        """
        lengths = {ep["name"]: ep["keys"].shape[0] for ep in pace.store.episodes[start:]}
        rows = [b for b, info in enumerate(infos) if info.get("stored_as") in lengths]
        names = [infos[b]["stored_as"] for b in rows]
        frames = recorder.episodes(rows, [lengths[n] for n in names])
        path = saved.with_name(saved.name.replace("rollouts_", "frames_", 1))
        save_frames(path, names, frames, [metas[b]["prompt"] for b in rows], sync=self.sync)
        for b in rows:
            infos[b]["frames_file"] = path.name

    def __call__(self, *a, tasks=None, shown=None, row_extra=None, names=None, runtime_seeds=None, save_tag=None,
                 recorder=None, **k):
        args = self.args
        with self.lock:
            pace = self.pace_getter()
            env = k["env"] if "env" in k else a[0]
            seeds = k.get("seeds")
            true_tasks = list(tasks) if tasks is not None else list(env.call("task_description"))
            task_keys = [normalize_instruction(t) for t in true_tasks]
            if shown is None and self.prompts:
                shown = [self.prompts(t) for t in true_tasks]
            episode_ids = []
            for key in task_keys:  # per-task episode counter: env i of rollout batch r = episode r * n_envs + i
                episode_ids.append(self.episodes.get(key, 0))
                self.episodes[key] = episode_ids[-1] + 1
            if runtime_seeds is None:
                runtime_seeds = [(zlib.crc32(key.encode()) + 7919 * e + args.seed) % (2 ** 31)
                                 for key, e in zip(task_keys, episode_ids)]
            stored_before = len(pace.store) if pace.store is not None else 0
            if pace.retrieval == "v3":
                # the retrieval knows what the policy is told (its rollouts are stored under it), not the env's task
                told = [normalize_instruction(s) for s in (shown if shown is not None else true_tasks)]
                texts = {"shown": told, "true": task_keys, "none": [None] * len(told)}[args.retrieval_text]
                keys = task_keys if args.retrieval_text == "true" else told
                pace.begin(keys, args.condition, seeds=runtime_seeds, true_tasks=task_keys, query_texts=texts)
            else:
                pace.begin(task_keys, args.condition, seeds=runtime_seeds)
            if shown is not None:
                if "env_preprocessor" in k:
                    k["env_preprocessor"] = FixedTasks(k["env_preprocessor"], shown)
                else:
                    a = (a[0], a[1], FixedTasks(a[2], shown), *a[3:])
            started = time.time()
            try:
                ret = self.original(*a, **k)
            except BaseException:
                pace.runtimes, pace.encoded = None, None
                raise
            done = ret["done"].bool()
            successes = ret["success"].bool().any(dim=1).tolist()
            T = done.shape[1]
            last = [int(torch.argmax(row.int())) if bool(row.any()) else T - 1 for row in done]
            n_steps = pace.policy.config.n_action_steps
            num_calls = [math.ceil((d + 1) / n_steps) for d in last]
            if names is None:
                names = [f"rollout/{args.condition}/{key[:60]}/ep{e:04d}" for key, e in zip(task_keys, episode_ids)]
            metas = [{"task": t, "episode": e, "seed": None if seeds is None else int(seeds[i]),
                      "prompt": shown[i] if shown is not None else t, "condition": args.condition,
                      **(row_extra[i] if row_extra else {})}
                     for i, (t, e) in enumerate(zip(true_tasks, episode_ids))]
            infos = pace.end(successes, num_calls, add_to_store=args.accumulate, names=names, metas=metas)
            if args.accumulate:
                with exclusive_writes(getattr(args, "write_lock", None), getattr(args, "write_gap", 0.0)):  # one at a time
                    saved = self.save_rollouts(pace, stored_before, save_tag or f"b{self.batches:05d}")
                    if recorder is not None and saved is not None:
                        self.save_frames(pace, stored_before, infos, metas, recorder, saved)
                if args.condition == "none" and saved is not None:
                    # no references in this run: nothing reads the stored rollouts back (the store is only saved), so
                    # keep the saved episodes without their image tokens (~130 KB per call) and the memory of a long
                    # collection stays flat
                    for i in range(stored_before, len(pace.store)):
                        pace.store.episodes[i] = {k: v for k, v in pace.store.episodes[i].items() if k != "vis"}
                for info in infos:
                    info["store_size"] = len(pace.store)
                    if saved is not None:
                        info["rollout_file"] = saved.name
            self.batches += 1
            rows = []
            for i, info in enumerate(infos):
                # num_calls: this episode's own calls (what its stored rollout keeps); the runtime counts the batch's
                # calls (the policy runs for every env until the batch's last episode ends): policy_calls
                row = {**metas[i], **(row_extra[i] if row_extra else {}), "success": bool(successes[i]),
                       "env_steps": last[i] + 1, "rollout_s": time.time() - started,
                       **{k: v for k, v in info.items() if k != "num_calls"},
                       "num_calls": num_calls[i], "policy_calls": info.get("num_calls")}
                rows.append(row)
                self.log.write(json.dumps(row) + "\n")
            self.log.flush()
            self.last_rows = rows
            return ret


def main() -> None:
    args, rest = parse_args(sys.argv[1:])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pace_args.json", "w") as f:
        json.dump({**vars(args), "lerobot_eval_args": rest}, f, indent=2)
    sys.argv = [sys.argv[0], *rest]

    import lerobot.scripts.lerobot_eval as lerobot_eval

    prompts = PromptMap(args.prompt_override, args.prompt_map)
    state = {"pace": None}
    log = open(out_dir / "pace_episodes.jsonl", "a")
    original_make_policy = lerobot_eval.make_policy

    def make_policy(*a, **k):
        policy_cfg = k["cfg"] if "cfg" in k else a[0]
        if policy_cfg.compile_model:
            raise ValueError("PACE needs --policy.compile_model=false (a compiled sample_actions bypasses its hooks)")
        policy = original_make_policy(*a, **k)
        if policy_cfg.pretrained_path:
            check_loaded(policy, str(policy_cfg.pretrained_path))
        state["pace"] = attach_pace(policy, args)
        return policy

    lerobot_eval.make_policy = make_policy
    rollout_dir = out_dir / "rollouts" if args.accumulate else None
    if rollout_dir is not None and rollout_files(rollout_dir):
        raise SystemExit(f"{rollout_dir} already holds rollouts: use a new --pace.out_dir (lerobot-eval cannot resume; "
                         "pace.eval_suite can)")
    lerobot_eval.rollout = PaceRollout(lerobot_eval.rollout, lambda: state["pace"], args, prompts, log, rollout_dir)
    try:
        lerobot_eval.main()
    finally:
        log.close()
        pace = state["pace"]
        if args.accumulate and pace is not None and pace.store is not None:
            print(f"[pace] the store ended with {len(pace.store)} episodes; this run's rollouts are in {rollout_dir} "
                  "(the library plus those files rebuild it)", flush=True)


if __name__ == "__main__":
    main()
