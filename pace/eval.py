"""
`lerobot-eval` with PACE attached to the pi0.5 policy. Every `--pace.*` option is read here; everything else goes to
lerobot-eval unchanged (env, policy, seeds, videos, output dir). Two lerobot-eval functions are wrapped:
  - `make_policy`: the created policy gets a PI05Experience (adapter + experience store);
  - `rollout`: before a batch of episodes, one runtime per env is started with the env's true instruction as task
    key (the policy may be told something else: --pace.prompt_override / --pace.prompt_map); after it, the episodes
    are closed, one json line per episode goes to <out_dir>/pace_episodes.jsonl and, with --pace.accumulate, the
    rollouts join the store, so later episodes can retrieve earlier ones (cross-episode experience).
Rollouts are serialized (a lock), since the episode state lives on the shared policy.

Episodes are paired across runs by (task, episode): LIBERO env i of rollout batch r starts from init state
r * batch_size + i and lerobot-eval seeds it with seed + r * batch_size + i, so keep --eval.batch_size and --seed
equal across the conditions being compared.

Without --pace.adapter a zero-initialized adapter is attached: its output is exactly the base policy's (a check of
the attachment itself; compare with plain lerobot-eval).

    python -m pace.eval --pace.adapter runs/goal_main/adapter_last.pt --pace.library libs/libero_goal_demo.pt \
        --pace.condition real --pace.out_dir eval/goal_real -- \
        --env.type=libero --env.task=libero_goal --eval.batch_size=10 --eval.n_episodes=50 \
        --policy.path=lerobot/pi05_libero_finetuned --policy.n_action_steps=10 --output_dir=eval/goal_real/lerobot
"""

import argparse
import json
import math
import sys
import threading
import time
import zlib
from pathlib import Path

import torch

from .common import PromptMap
from .core import CONDITIONS, ExperienceStore, load_adapter, normalize_instruction
from .pi05 import PI05Experience, build_adapter


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="python -m pace.eval", add_help=True)
    parser.add_argument("--pace.adapter", dest="adapter", default=None)
    parser.add_argument("--pace.library", dest="library", nargs="*", default=[], help="Store files (demo libraries)")
    parser.add_argument("--pace.condition", dest="condition", choices=CONDITIONS, default="real")
    parser.add_argument("--pace.out_dir", dest="out_dir", required=True)
    parser.add_argument("--pace.num_refs", dest="num_refs", type=int, default=4)
    parser.add_argument("--pace.pool_size", dest="pool_size", type=int, default=64)
    parser.add_argument("--pace.select", dest="select", choices=("similar", "recent", "random"), default="similar")
    parser.add_argument("--pace.sources", dest="sources", nargs="+", default=["demo", "rollout"])
    parser.add_argument("--pace.include_failures", dest="include_failures", action="store_true",
                        help="Failed episodes may be references too (their success flag is an adapter input)")
    parser.add_argument("--pace.vmax", dest="vmax", type=int, default=None, help="Default: the adapter's training value")
    parser.add_argument("--pace.gamma", dest="gamma", type=float, default=None)
    parser.add_argument("--pace.cameras", dest="cameras", type=int, nargs="+", default=None,
                        help="Default: the store's (0 1 without a store)")
    parser.add_argument("--pace.accumulate", dest="accumulate", action="store_true",
                        help="Add every finished episode to the store (saved to <out_dir>/pace_store.pt)")
    parser.add_argument("--pace.prompt_override", dest="prompt_override", default=None)
    parser.add_argument("--pace.prompt_map", dest="prompt_map", default=None)
    parser.add_argument("--pace.seed", dest="seed", type=int, default=0, help="Runtime RNG (control conditions)")
    if "--" in argv:
        split = argv.index("--")
        return parser.parse_args(argv[:split]), argv[split + 1 :]
    return parser.parse_known_args(argv)


class TaskOverride:
    """The env preprocessor, with the instructions the policy sees replaced (the env keeps the true task)."""

    def __init__(self, inner, prompts: PromptMap):
        self.inner, self.prompts = inner, prompts

    def __call__(self, observation):
        observation = dict(observation)
        observation["task"] = [self.prompts(t) for t in observation["task"]]
        return self.inner(observation)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main() -> None:
    args, rest = parse_args(sys.argv[1:])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pace_args.json", "w") as f:
        json.dump({**vars(args), "lerobot_eval_args": rest}, f, indent=2)
    sys.argv = [sys.argv[0], *rest]

    import lerobot.scripts.lerobot_eval as lerobot_eval

    prompts = PromptMap(args.prompt_override, args.prompt_map)
    state = {"pace": None, "episodes": {}}
    lock = threading.Lock()
    log = open(out_dir / "pace_episodes.jsonl", "a")
    original_make_policy, original_rollout = lerobot_eval.make_policy, lerobot_eval.rollout

    def make_policy(*a, **k):
        policy = original_make_policy(*a, **k)
        store = ExperienceStore.load(args.library) if args.library else None
        if args.adapter:
            adapter, ckpt = load_adapter(args.adapter)
            trained = ckpt.get("args", {})
        else:
            adapter, trained = build_adapter(policy), {}
        info = store.info if store is not None else {}
        cameras = args.cameras or info.get("cameras") or trained.get("cameras") or (0, 1)
        runtime_kwargs = dict(
            num_refs=args.num_refs, pool_size=args.pool_size, select=args.select, sources=tuple(args.sources),
            success_only=not args.include_failures,
            vmax=args.vmax if args.vmax is not None else int(trained.get("vmax", 3)),
            gamma=args.gamma if args.gamma is not None else float(trained.get("gamma", 0.01)),
        )
        if store is not None and store.info.get("cameras") and list(cameras) != list(store.info["cameras"]):
            raise ValueError(f"--pace.cameras {cameras} vs the store's {store.info['cameras']}")
        pace = PI05Experience(policy, adapter, store, cameras=cameras, sig_grid=info.get("sig_grid", 4),
                              runtime_kwargs=runtime_kwargs)
        pace.adapter.eval()
        state["pace"] = pace
        print(f"[pace] condition={args.condition} adapter={args.adapter or 'zero-init'} "
              f"store={0 if store is None else len(store)} episodes runtime={runtime_kwargs}", flush=True)
        return policy

    def rollout(*a, **k):
        with lock:
            pace = state["pace"]
            env = k["env"] if "env" in k else a[0]
            seeds = k.get("seeds")
            true_tasks = list(env.call("task_description"))
            task_keys = [normalize_instruction(t) for t in true_tasks]
            episode_ids = []
            for key in task_keys:  # per-task episode counter: env i of rollout batch r = episode r * n_envs + i
                episode_ids.append(state["episodes"].get(key, 0))
                state["episodes"][key] = episode_ids[-1] + 1
            runtime_seeds = [(zlib.crc32(key.encode()) + 7919 * e + args.seed) % (2 ** 31)
                             for key, e in zip(task_keys, episode_ids)]
            pace.begin(task_keys, args.condition, seeds=runtime_seeds)
            if prompts:
                if "env_preprocessor" in k:
                    k["env_preprocessor"] = TaskOverride(k["env_preprocessor"], prompts)
                else:
                    a = (a[0], a[1], TaskOverride(a[2], prompts), *a[3:])
            started = time.time()
            try:
                ret = original_rollout(*a, **k)
            except BaseException:
                pace.runtimes, pace.encoded = None, None
                raise
            done = ret["done"].bool()
            successes = ret["success"].bool().any(dim=1).tolist()
            T = done.shape[1]
            last = [int(torch.argmax(row.int())) if bool(row.any()) else T - 1 for row in done]
            n_steps = pace.policy.config.n_action_steps
            num_calls = [math.ceil((d + 1) / n_steps) for d in last]
            names = [f"rollout/{args.condition}/{key[:60]}/ep{e:04d}" for key, e in zip(task_keys, episode_ids)]
            metas = [{"task": t, "episode": e, "seed": None if seeds is None else int(seeds[i]),
                      "prompt": prompts(t) if prompts else t, "condition": args.condition}
                     for i, (t, e) in enumerate(zip(true_tasks, episode_ids))]
            infos = pace.end(successes, num_calls, add_to_store=args.accumulate, names=names, metas=metas)
            for i, info in enumerate(infos):
                row = {**metas[i], "success": bool(successes[i]), "env_steps": last[i] + 1,
                       "num_calls": num_calls[i], "rollout_s": time.time() - started, **info}
                log.write(json.dumps(row) + "\n")
            log.flush()
            return ret

    lerobot_eval.make_policy = make_policy
    lerobot_eval.rollout = rollout
    try:
        lerobot_eval.main()
    finally:
        log.close()
        pace = state["pace"]
        if args.accumulate and pace is not None and pace.store is not None:
            pace.store.save(out_dir / "pace_store.pt")
            print(f"[pace] saved {len(pace.store)} episodes to {out_dir / 'pace_store.pt'}", flush=True)


if __name__ == "__main__":
    main()
