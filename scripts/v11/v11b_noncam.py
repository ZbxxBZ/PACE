"""
V11b training data on non-camera LIBERO-plus variants (11-01; desktop design doc 10.14f): pi0.5's own rollouts, Delta* = 0. The
variants are drawn at random (seed 11) from the ones outside the 30 non-camera test variants and the reference sources: 40 robot
initial states, 15 object layouts, 15 sensor noise, 10 background, 10 language, 10 light for training at init states 6-44; 4 + 2 x 5
other variants for validation at init states 45-49; three init states each. Frames are the env's raw observations, so sensor
noise is in them (a replay would miss it). pi0.5 gets the variant's own instruction, as in the non-camera test.
Same formats as V11's data: <out>/frames/noncam/<key>.npz, <out>/api/noncam/<key>.npy, <out>/noncam.jsonl. Resumable.
    python v11b_noncam.py --out runs/v11b
"""
import argparse
import collections
import functools
import json
import os
import time

import numpy as np
import torch

import closed_loop_v11 as C
import v10_oracle_shift2 as O
from v11_check import CAMS, load_pi05
from v11b_recenv import rec_env

ROOT = C.ROOT
N_TRAIN = {"Robot Initial States": 40, "Objects Layout": 15, "Sensor Noise": 15, "Background Textures": 10,
           "Language Instructions": 10, "Light Conditions": 10}
N_VAL = {"Robot Initial States": 4, "Objects Layout": 2, "Sensor Noise": 2, "Background Textures": 2,
         "Language Instructions": 2, "Light Conditions": 2}


def entries():
    F = json.load(open(f"{ROOT}/rectest/v10_shift/full_targets.json"))
    test = {x["task_id"] for x in json.load(open(f"{ROOT}/rectest/v10_shift/noncam_targets.json"))}
    src = {json.loads(l)["task_id"] for l in open(f"{ROOT}/v10b/noise_sources/episodes.jsonl")}
    rng = np.random.default_rng(11)
    out = []
    for cat in N_TRAIN:
        pool = sorted(x["task_id"] for x in F if x["category"] == cat and x["task_id"] not in test | src)
        pick = [pool[i] for i in rng.permutation(len(pool))[:N_TRAIN[cat] + N_VAL[cat]]]
        by = {x["task_id"]: x for x in F}
        for j, tid in enumerate(pick):
            val = j >= N_TRAIN[cat]
            inits = rng.choice(np.arange(45, 50) if val else np.arange(6, 45), 3, replace=False)
            for k in sorted(int(i) for i in inits):
                x = by[tid]
                out.append(dict(key=f"t{tid}_i{k}", task_id=tid, init=k, seed=1000 + 7 * tid + k, category=cat, base=x["base"],
                                language=x["language"], split="val" if val else "train"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/pace/runs/v11b")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke: the first episodes only")
    args = ap.parse_args()
    for d in ("frames/noncam", "api/noncam"):
        os.makedirs(f"{args.out}/{d}", exist_ok=True)
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import _LazyAsyncVectorEnv
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts.lerobot_eval import rollout
    from pace.eval_suite import FixedTasks

    O.worker_init()
    policy, noise, pre, env_pre, _, _ = load_pi05()
    _, post = make_pre_post_processors(policy.config, pretrained_path="lerobot/pi05_libero_finetuned",
                                       preprocessor_overrides={"device_processor": {"device": "cuda"},
                                                               "rename_observations_processor": {"rename_map": {}}})
    _, env_post = make_env_pre_post_processors(O.G["env_cfg"], policy.config)
    E = entries()
    print("[noncam] variants", collections.Counter((e["category"], e["split"]) for e in E[::3]), f"{len(E)} episodes", flush=True)
    log_path = f"{args.out}/noncam.jsonl"
    done = {json.loads(l)["key"] for l in open(log_path)} if os.path.exists(log_path) else set()
    todo = [e for e in E if e["key"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    S = dict(a=None)
    orig = policy.predict_action_chunk

    def pred(batch, **kw):
        act = orig(batch, **kw)
        S["a"].append(act.float().cpu().numpy())
        return act

    policy.predict_action_chunk = pred
    log = open(log_path, "a")
    t0, agent, wrist = time.time(), CAMS["agentview_image"], CAMS["robot0_eye_in_hand_image"]
    for b0 in range(0, len(todo), args.batch):
        batch = todo[b0:b0 + args.batch]
        fns = []
        for e in batch:
            fn = C.make_env(e["task_id"], e["init"])
            fns.append(functools.partial(rec_env, fn.func, e["init"], f"{args.out}/frames/noncam/{e['key']}.npz", agent, wrist, **fn.keywords))
        S["a"] = []
        noise.set([e["seed"] for e in batch])
        env = _LazyAsyncVectorEnv(fns)
        try:
            torch.manual_seed(batch[0]["seed"])
            ret = rollout(env=env, policy=policy, env_preprocessor=FixedTasks(env_pre, [e["language"] for e in batch]),
                          env_postprocessor=env_post, preprocessor=pre, postprocessor=post, seeds=[e["seed"] for e in batch])
            env.call("flush")
        finally:
            env.close()
        succ = ret["success"].bool().any(dim=1).tolist()
        A = np.stack(S["a"], 1)  # [rows, calls, 50, 7]
        for i, e in enumerate(batch):
            f = f"{args.out}/frames/noncam/{e['key']}.npz"
            with np.load(f) as z:
                T, rows, ok = z["t"], len(z["t"]), bool(z["success"])
            assert ok == bool(succ[i]) and np.array_equal(T, np.arange(rows) * C.CALL), (e["key"], ok, succ[i], T[:4])
            np.save(f"{args.out}/api/noncam/{e['key']}.npy", A[i, :rows].astype(np.float16))
            log.write(json.dumps(dict(e, success=bool(succ[i]), calls=rows, gpu=torch.cuda.get_device_name(0))) + "\n")
        log.flush()
        print(f"[noncam] {b0 + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s), batch success {sum(succ)}/{len(batch)}", flush=True)
    print(f"-> {args.out} noncam", flush=True)


if __name__ == "__main__":
    main()
