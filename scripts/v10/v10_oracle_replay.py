"""
V10 premise 2 (10-21): the ceiling of transferring a success across visual perturbations. In a camera-viewpoint
variant's env, replay open-loop the executed actions of successes recorded on OTHER variants of the same base task
with the same physical setup (background, light, sensor noise, language: same base init state, same robot), and see
whether the task succeeds. No policy runs (only its processors, to unnormalize the recorded actions as the closed loop
did), so the camera cannot matter: success here means the setup is physically the same and the recorded actions solve
it. Controls: each source replayed in its own variant (must succeed: the replay is exact) and in the base task's
objects-layout variants (other layouts: should mostly fail).
Sources: V7's closed-loop runs on the evaluation's 140 variants (plus_l10_V7_5500_acc3, then _other_acc3), successes
only, up to --per_target per target from distinct source variants.
    python v10_oracle_replay.py --out /root/pace/runs/rectest/v10_oracle_replay.json
"""
import argparse
import collections
import glob
import json
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--per_target", type=int, default=3)
    parser.add_argument("--out", default="/root/pace/runs/rectest/v10_oracle_replay.json")
    args = parser.parse_args()

    RUNS = ["/root/pace/runs/eval/plus_l10_V7_5500_acc3", "/root/pace/runs/eval/plus_l10_V7_5500_other_acc3"]
    SAME = ("Background Textures", "Light Conditions", "Sensor Noise", "Language Instructions")

    eps = []
    for run in RUNS:
        for f in sorted(glob.glob(f"{run}/rollouts/rollouts_*.pt")):
            for e in torch.load(f, map_location="cpu", weights_only=False)["episodes"]:
                m = e["meta"]
                eps.append(dict(task_id=int(m["task_id"]), cat=m["category"], base=m["base"], round=int(m["round"]),
                                seed=int(m["seed"]), init_index=int(m.get("init_index", 0)), success=bool(e["success"]),
                                actions=e["actions"].float(), name=e["name"], run=run.rsplit("/", 1)[-1]))
    variants = {}
    for e in eps:
        variants.setdefault(e["task_id"], e)  # first record: its round-1 seed and meta
    succ_by_base = collections.defaultdict(list)
    for e in eps:
        if e["success"] and e["cat"] in SAME:
            succ_by_base[e["base"]].append(e)
    for b in succ_by_base:
        succ_by_base[b].sort(key=lambda e: (e["task_id"], e["round"], e["run"]))


    def pick_sources(base, n):
        out, seen = [], set()
        for e in succ_by_base[base]:
            if e["task_id"] not in seen:
                out.append(e)
                seen.add(e["task_id"])
            if len(out) == n:
                break
        return out


    jobs = []  # (kind, target variant meta, [source episodes])
    for t, v in sorted(variants.items()):
        if v["cat"] == "Camera Viewpoints":
            jobs.append(("camera", v, pick_sources(v["base"], args.per_target)))
    for t, v in sorted(variants.items()):
        if v["cat"] == "Objects Layout":
            jobs.append(("objects_layout", v, pick_sources(v["base"], args.per_target)))
    own = {}
    for e in (s for b in sorted(succ_by_base) for s in pick_sources(b, 1)):
        own[e["task_id"]] = e
    for t, e in sorted(own.items()):
        jobs.append(("own_variant", variants[t] | {"seed": e["seed"]}, [e]))
    print(f"{len(eps)} recorded episodes; jobs: " + ", ".join(f"{k} {n}" for k, n in
          collections.Counter(j[0] for j in jobs).items()), flush=True)

    from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
    from lerobot.envs.configs import LiberoPlusEnv  # noqa: E402
    from lerobot.envs.factory import make_env_pre_post_processors  # noqa: E402
    from lerobot.envs.libero import _get_suite, _LazyAsyncVectorEnv, _make_env_fns, parse_camera_names  # noqa: E402
    from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
    from lerobot.scripts import lerobot_eval  # noqa: E402
    from lerobot.utils.constants import ACTION  # noqa: E402

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.n_action_steps = 10
    _, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}, "rename_observations_processor": {"rename_map": {}}})
    env_cfg = LiberoPlusEnv(task=args.suite)
    _, env_post = make_env_pre_post_processors(env_cfg, cfg)
    suite = _get_suite(args.suite)
    gym_kwargs = dict(env_cfg.gym_kwargs)
    gym_kwargs.pop("task_ids", None)
    camera_names = parse_camera_names(env_cfg.camera_name)
    STEPS = 10  # executed steps per recorded call


    def make_env(task_id, init_index, K):
        fn = _make_env_fns(suite=suite, suite_name=args.suite, task_id=int(task_id), n_envs=1,
                           camera_names=camera_names, episode_length=env_cfg.episode_length,
                           init_states=env_cfg.init_states, gym_kwargs=gym_kwargs, control_mode=env_cfg.control_mode,
                           camera_name_mapping=env_cfg.camera_name_mapping, is_libero_plus=env_cfg.is_libero_plus)[0]
        if int(init_index):
            import functools
            fn = functools.partial(fn.func, int(init_index), **fn.keywords)
        return _LazyAsyncVectorEnv([fn] * K)


    def replay(target, sources):
        """Replay every source's recorded executed actions in its own copy of the target's env; success per source."""
        K = len(sources)
        env = make_env(target["task_id"], target["init_index"], K)
        try:
            env.reset(seed=[target["seed"]] * K, options={lerobot_eval.NEW_ROLLOUT_OPTION: True})
            postprocessor.reset()
            acts = [s["actions"][:, :STEPS].reshape(-1, s["actions"].shape[-1]) for s in sources]  # [calls*10, A]
            n = max(len(a) for a in acts)
            max_steps = env.call("_max_episode_steps")[0]
            done, success, end = np.zeros(K, bool), np.zeros(K, bool), np.full(K, -1)
            for t in range(min(n, max_steps)):
                batch = torch.stack([a[min(t, len(a) - 1)] for a in acts])
                a = env_post({ACTION: postprocessor(batch)})[ACTION]
                a = a.cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
                _, _, terminated, truncated, info = env.step(a)
                ok = np.zeros(K, bool)
                fin = info.get("final_info")
                if isinstance(fin, dict) and "is_success" in fin:
                    ok = np.asarray(fin["is_success"], bool).reshape(-1)
                elif "is_success" in info:
                    ok = np.asarray(info["is_success"], bool).reshape(-1)
                ended = np.asarray(terminated, bool) | np.asarray(truncated, bool)
                for b in range(K):
                    if not done[b]:
                        if ok[b]:
                            success[b] = True
                        # a source that ran out of actions without success counts as a failure
                        if ok[b] or ended[b] or t >= len(acts[b]) - 1:
                            done[b], end[b] = True, t + 1
                if done.all():
                    break
            return success.tolist(), end.tolist()
        finally:
            env.close()


    results, t0 = [], time.time()
    for k, (kind, target, sources) in enumerate(jobs):
        if not sources:
            continue
        ok, end = replay(target, sources)
        results.append(dict(kind=kind, target=target["task_id"], target_cat=target["cat"], base=target["base"],
                            sources=[s["task_id"] for s in sources], source_cats=[s["cat"] for s in sources],
                            source_steps=[int(s["actions"].shape[0]) * STEPS for s in sources], success=ok, end=end))
        print(f"[replay] {k + 1}/{len(jobs)} {kind:14s} target {target['task_id']:5d} ({target['base'][:40]}): "
              f"{sum(ok)}/{len(ok)} from {[s['task_id'] for s in sources]}  ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(args.out, "w"), indent=1)

    print("\n[summary] replayed successes that succeed again:")
    for kind in ("own_variant", "camera", "objects_layout"):
        rs = [r for r in results if r["kind"] == kind]
        n = sum(len(r["success"]) for r in rs)
        s = sum(sum(r["success"]) for r in rs)
        tgt = sum(any(r["success"]) for r in rs)
        print(f"  {kind:14s} {s}/{n} replays ({100 * s / max(n, 1):.0f}%); targets with any success {tgt}/{len(rs)}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
