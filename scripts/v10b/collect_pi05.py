"""
pi0.5 rollouts in LIBERO-plus's code base that keep only what a later exact replay needs (10-23, V10-1b data): per
episode the executed env-unit actions (what lerobot's rollout hands to env.step), success, steps, task, init state,
seed. Images are not stored: the simulator is deterministic, so any frame (any camera, any augmentation) is rendered
again by replaying the actions from the same init state and seed.
  --set noise_sources : sources for the d-error test (v10_oracle_noise.py) -- for each base task of the 20 camera
                        targets (runs/rectest/v10_shift/base_targets), its first --per_base light / sensor-noise /
                        background variants from init state 0;
  --set orig          : the V10-1b references -- the 10 ORIGINAL libero_10 tasks (LIBERO-plus's own bddl_files/
                        libero_10/<stem>.bddl and init_files/libero_10/<stem>.pruned_init, no perturbation), init states
                        --inits (default 6..49; 0 and 1..5 are test scenes). Their success also labels "pi0.5 alone".
  --set orig_aug      : the same scenes, each under --n_aug of its augmentations (augment.scene_aug, the ones its pair
                        renders use): "pi0.5 alone" under what the network sees, for the alpha labels (10-24).
GPU policy, batches of --batch envs (lerobot's vector env), EGL rendering. Resumable (skips recorded episodes).
    python collect_pi05.py --set orig --out runs/v10b/refs
"""
import argparse
import collections
import functools
import json
import os
import time

import numpy as np
import torch

import augment as A

OFFSET = 10000  # task ids of the original tasks inside the combined suite
CATS = ("Light Conditions", "Sensor Noise", "Background Textures")


class _Tasks:
    """Index access over both task lists (module level, so the suite pickles into the vector env's workers)."""

    def __init__(self, base, extra):
        self.base, self.extra = base, extra

    def __getitem__(self, i):
        return self.extra[i - OFFSET] if i >= OFFSET else self.base.tasks[i]

    def __len__(self):
        return len(self.base.tasks)


class ComboSuite:
    """LIBERO-plus libero_10 (ids < OFFSET) plus the 10 original tasks (ids OFFSET + i)."""

    def __init__(self, base, extra):
        self.base, self.extra = base, extra
        self.tasks = _Tasks(base, extra)

    def get_task(self, i):
        return self.tasks[i]

    def __getattr__(self, k):
        if k.startswith("__") or k in ("base", "extra", "tasks"):  # not set yet while unpickling
            raise AttributeError(k)
        return getattr(self.base, k)


def stems():
    import libero.libero as L

    d = os.path.join(os.path.dirname(L.__file__), "init_files", "libero_10")
    return sorted(f[:-len(".pruned_init")] for f in os.listdir(d) if f.endswith(".pruned_init"))


def stem_language(stem):
    parts = stem.split("_")
    i = next(i for i, p in enumerate(parts) if p.startswith("SCENE")) + 1
    return " ".join(parts[i:]).lower()


def original_tasks(base_suite):
    proto = base_suite.tasks[0]
    return [proto._replace(name=s, language=stem_language(s), problem_folder="libero_10", bddl_file=s + ".bddl",
                           init_states_file=s + ".pruned_init") for s in stems()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["noise_sources", "orig", "orig_aug"], required=True)
    ap.add_argument("--n_aug", type=int, default=3)
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(6, 50)))
    ap.add_argument("--per_base", type=int, default=3)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _get_suite, _LazyAsyncVectorEnv, _make_env_fns, parse_camera_names
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from lerobot.utils.random_utils import set_seed
    from pace.common import load_policy
    from pace.eval_suite import FixedTasks, classification

    env_cfg = LiberoPlusEnv(task="libero_10")
    base = _get_suite("libero_10")
    suite = ComboSuite(base, original_tasks(base))
    entries = []
    if args.set == "orig":
        for i, t in enumerate(suite.extra):
            for k in args.inits:
                entries.append(dict(task_id=OFFSET + i, name=t.name, base=t.name, language=t.language, init=k,
                                    category="Original"))
    elif args.set == "orig_aug":
        for i, t in enumerate(suite.extra):
            for k in args.inits:
                for s_ in range(args.n_aug):
                    entries.append(dict(task_id=OFFSET + i, name=t.name, base=t.name, language=t.language, init=k,
                                        category="Original", aug_idx=s_, aug=A.scene_aug(OFFSET + i, k, s_)))
    else:
        targets = [json.loads(l) for l in open("/root/pace/runs/rectest/v10_shift/base_targets/episodes.jsonl")]
        want = sorted({r["name"].split("_view_")[0] for r in targets})
        cls = classification("libero_10")
        st = stems()
        by = collections.defaultdict(list)
        for tid in range(base.n_tasks):
            t = base.get_task(tid)
            cat = cls.get(t.name, (None,))[0]
            b = max((s for s in st if t.name.startswith(s)), key=len, default=None)
            if cat in CATS and b in want:
                by[b].append((CATS.index(cat), tid, t.name, cat))
        for b in want:
            for _, tid, name, cat in sorted(by[b])[:args.per_base]:
                entries.append(dict(task_id=tid, name=name, base=b, language=stem_language(b), init=0, category=cat))
    for j, e in enumerate(entries):
        e["seed"] = args.seed + j
    log_path = f"{args.out}/episodes.jsonl"
    ek = lambda e: (e["task_id"], e["init"], e.get("aug_idx"))
    done = {ek(r) for r in map(json.loads, open(log_path))} if os.path.exists(log_path) else set()
    todo = [e for e in entries if ek(e) not in done]
    print(f"[collect] {args.set}: {len(entries)} episodes, {len(todo)} to do, batch {args.batch}", flush=True)

    set_seed(args.seed)
    policy, _ = load_policy("lerobot/pi05_libero_finetuned", "cuda", n_action_steps=10)
    pre, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                         preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                                 "rename_observations_processor": {"rename_map": {}}})
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    gk = dict(env_cfg.gym_kwargs)
    gk.pop("task_ids", None)
    cams = parse_camera_names(env_cfg.camera_name)
    log = open(log_path, "a")
    t0 = time.time()
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        fns = []
        for e in batch:
            fn = _make_env_fns(suite=suite, suite_name="libero_10", task_id=e["task_id"], n_envs=1, camera_names=cams,
                               episode_length=env_cfg.episode_length, init_states=env_cfg.init_states, gym_kwargs=gk,
                               control_mode=env_cfg.control_mode, camera_name_mapping=env_cfg.camera_name_mapping,
                               is_libero_plus=env_cfg.is_libero_plus)[0]
            if "aug" in e:
                fns.append(functools.partial(A.aug_env, fn.func, e["init"], e["aug"], **fn.keywords))
            else:
                fns.append(functools.partial(fn.func, e["init"], **fn.keywords))
        env = _LazyAsyncVectorEnv(fns)
        try:
            torch.manual_seed(batch[0]["seed"])
            ret = rollout(env=env, policy=policy, env_preprocessor=FixedTasks(env_pre, [e["language"] for e in batch]),
                          env_postprocessor=env_post, preprocessor=pre, postprocessor=post,
                          seeds=[e["seed"] for e in batch])
        finally:
            env.close()
        dn = ret["done"].bool()
        succ = ret["success"].bool().any(dim=1).tolist()
        acts = ret["action"].float().cpu().numpy()
        for i, e in enumerate(batch):
            T = int(torch.argmax(dn[i].int())) + 1 if bool(dn[i].any()) else dn.shape[1]
            key = f"t{e['task_id']}_i{e['init']}" + (f"_a{e['aug_idx']}" if "aug_idx" in e else "")
            np.save(f"{args.out}/{key}.npy", acts[i, :T].astype(np.float32))
            log.write(json.dumps({**e, "success": bool(succ[i]), "steps": T, "actions": f"{key}.npy",
                                  "gpu": torch.cuda.get_device_name(0)}) + "\n")
        log.flush()
        n = b0 + len(batch)
        print(f"[collect] {n}/{len(todo)} ({time.time() - t0:.0f}s), batch success {sum(succ)}/{len(batch)}", flush=True)
    rows = [json.loads(l) for l in open(log_path)]
    print(f"[collect] done: {sum(r['success'] for r in rows)}/{len(rows)} succeed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
