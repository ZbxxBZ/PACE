"""
V10 reference pick on all 7 LIBERO-plus categories (10-23, registered in docs/PACE_V1-V8.md "选参考：扩展到全部 7 类").
v10_key_pca.py covered the 4 visual categories (800 frames in runs/rectest/v10_keypca/frames.pt); this adds per base task
2 variants of Robot Initial States, Objects Layout and Language Instructions x init states 0..9 (600 frames), rendered
and embedded the same way (closed-loop preprocessing, pi0.5's frozen image tokens, GPU), plus per frame: the movable
objects' xy, the arm's joint positions, and for robot-init frames a "canonical" frame -- the arm set to the default
layout's joints of the same init state (from its .pruned_init) before rendering, i.e. "move to the reference's start
pose, then look" (the objects are not touched).
Query: every frame (1400); candidates: the same base task's frames of other variants. Methods (cosine of the flattened
unit cells): agent / wrist / both; wrist-canon (robot-init queries use the canonical frame); weighted
(w * wrist + (1 - w) * agent, w = exp(-|dq_arm| / 0.05 rad) between query and candidate). Layout distance = largest xy
displacement over the movable objects both frames have. Metrics per query category: same-layout hit of the top-1
(a candidate within 5 mm exists), top-1 distance; without same-layout candidates: top-1 / random / best.
Instruction check: rank of the correct base task among the 10 base instructions by pace.pi05.InstructionText (fit on
the 10 base instructions), for the shown instruction of every frame (language variants: the paraphrase).
    python v10_key_all.py --workers 24 --device cuda -> runs/rectest/v10_keypca/{frames_all.pt,result_all.json}
"""
import argparse
import collections
import functools
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np
import torch

import v10_key_pca as K

OUT = K.OUT
NEW = ("Robot Initial States", "Objects Layout", "Language Instructions")


def batched(x):
    if isinstance(x, np.ndarray):
        return x[None]
    if isinstance(x, dict):
        return {k: batched(v) for k, v in x.items()}
    return x


def render(job):
    import gymnasium as gym
    from lerobot.envs.utils import preprocess_observation

    tid, k, seed, default_arm = job
    c = K.G["cfg"]
    fn = K.G["make_fns"](suite=K.G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=K.G["cams"],
                         episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=K.G["gk"],
                         control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    env = gym.vector.SyncVectorEnv([functools.partial(fn.func, int(k), **fn.keywords)])
    try:
        obs, _ = env.reset(seed=[int(seed)])
        lib = env.envs[0]
        inner = lib._env.env
        robot = inner.robots[0]
        pos = {n: np.array(inner.sim.data.body_xpos[i], dtype=np.float64)[:2] for n, i in inner.obj_body_id.items()
               if n in inner.objects_dict}
        arm = np.array(inner.sim.data.qpos[robot._ref_joint_pos_indexes], dtype=np.float64)
        o = preprocess_observation(obs)
        o["task"] = list(env.call("task_description"))
        canon = None
        if default_arm is not None:
            inner.sim.data.qpos[robot._ref_joint_pos_indexes] = default_arm
            inner.sim.data.qvel[:] = 0
            inner.sim.forward()
            raw = inner._get_observations(force_update=True)
            co = preprocess_observation(batched(lib._format_raw_obs(raw)))
            co["task"] = o["task"]
            canon = K.to_numpy(co)
        return tid, k, K.to_numpy(o), canon, pos, arm, o["task"][0]
    finally:
        env.close()


def render_all(jobs, workers, rounds=5):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    done, todo = {}, list(jobs)
    for rnd in range(rounds):
        if not todo:
            break
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=K.worker_init) as ex:
            futs = {ex.submit(render, j): j for j in todo}
            try:
                for f in as_completed(futs):
                    r = f.result()
                    done[(r[0], r[1])] = r
                    if len(done) % 100 == 0:
                        print(f"[render] {len(done)}/{len(jobs)}", flush=True)
            except BrokenProcessPool as e:
                print(f"[render] pool broke in round {rnd} ({e}); {len(done)}/{len(jobs)} done", flush=True)
        todo = [j for j in jobs if (j[0], j[1]) not in done]
    assert not todo, f"{len(todo)} jobs never rendered"
    return list(done.values())


def pick_new(per_cat=2, seed=0):
    from pace.eval_suite import classification
    from lerobot.envs.libero import _get_suite
    import libero.libero as L

    suite = _get_suite("libero_10")
    cls = classification("libero_10")
    stems = sorted({f[:-len(".pruned_init")] for f in os.listdir(os.path.join(os.path.dirname(L.__file__), "init_files",
                                                                              "libero_10")) if f.endswith(".pruned_init")})
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid in range(suite.n_tasks):
        t = suite.get_task(tid)
        cat = cls.get(t.name, (None,))[0]
        base = max((s for s in stems if t.name.startswith(s)), key=len, default=None)
        if cat in NEW and base:
            by[base][cat].append(tid)
    rng = random.Random(seed)
    return {b: {c: sorted(rng.sample(sorted(d[c]), min(per_cat, len(d[c])))) for c in NEW} for b, d in sorted(by.items())}


def layout_dist(a, b):
    common = [n for n in a if n in b]
    return max(float(np.linalg.norm(a[n] - b[n])) for n in common) if common else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--per_cat", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()
    import libero.libero as L
    from lerobot.envs.libero import _get_suite

    suite = _get_suite("libero_10")
    picked = pick_new(args.per_cat)
    print("[variants] new categories:", {b[:40]: {c: len(v) for c, v in d.items()} for b, d in picked.items()}, flush=True)
    # the default layout's arm joints per (base, init), from its init file: flattened state = [time, qpos, qvel]
    K.worker_init()
    meta, jobs = {}, []
    idir = os.path.join(os.path.dirname(L.__file__), "init_files", "libero_10")
    arm_idx = None
    for base, d in picked.items():
        for cat, tids in d.items():
            for tid in tids:
                meta[tid] = (base, cat)
    # arm joint qpos indices: read once from a rendered env (same robot in every scene)
    import gymnasium as gym
    c = K.G["cfg"]
    any_tid = next(iter(meta))
    fn = K.G["make_fns"](suite=K.G["suite"], suite_name="libero_10", task_id=int(any_tid), n_envs=1,
                         camera_names=K.G["cams"], episode_length=c.episode_length, init_states=c.init_states,
                         gym_kwargs=K.G["gk"], control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                         is_libero_plus=c.is_libero_plus)[0]
    e = gym.vector.SyncVectorEnv([functools.partial(fn.func, 0, **fn.keywords)])
    e.reset(seed=[0])
    arm_idx = list(e.envs[0]._env.env.robots[0]._ref_joint_pos_indexes)
    e.close()
    defaults = {}
    for base in picked:
        st = torch.load(os.path.join(idir, base + ".pruned_init"), weights_only=False)
        defaults[base] = {k: np.array(st[k][1:][arm_idx], dtype=np.float64) for k in args.inits}
    for tid, (base, cat) in sorted(meta.items()):
        for k in args.inits:
            jobs.append((tid, k, 1000 + k, defaults[base][k] if cat == "Robot Initial States" else None))
    # the visual categories: image tokens from frames.pt, object positions from its layouts (one physical scene per
    # (task, init) in every visual variant), arm joints = the default layout's (their init file)
    old = torch.load(f"{OUT}/frames.pt", weights_only=False)
    vis_meta = {f["tid"]: (f["base"], f["cat"]) for f in old["frames"]}
    torch.multiprocessing.set_sharing_strategy("file_system")
    rendered = render_all(jobs, args.workers)
    print(f"[render] {len(rendered)} ({time.time() - t0:.0f}s)", flush=True)

    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from pace.common import load_policy
    from pace.pi05 import InstructionText, call_keys

    policy, preprocessor = load_policy("lerobot/pi05_libero_finetuned", args.device, n_action_steps=10)
    env_pre, _ = make_env_pre_post_processors(LiberoPlusEnv(task="libero_10"), policy.config)

    def embed(o):
        batch = preprocessor(env_pre(K.to_torch(o)))
        return call_keys(policy, batch, cameras=(0, 1), sig_grid=2, features=True)[2][0].half().cpu()

    frames = []
    for f in old["frames"]:
        lay = old["layouts"][f["base"]][f["init"]]
        frames.append(dict(base=f["base"], cat=f["cat"], tid=f["tid"], init=f["init"], vis=f["vis"], canon=f["vis"],
                           pos={n: np.array(v[:2], dtype=np.float64) for n, v in lay.items()},
                           arm=defaults[f["base"]][f["init"]], shown=""))
    for tid, k, o, canon, pos, arm, shown in sorted(rendered, key=lambda x: (x[0], x[1])):
        base, cat = meta[tid]
        vis = embed(o)
        frames.append(dict(base=base, cat=cat, tid=tid, init=k, vis=vis, canon=embed(canon) if canon is not None else vis,
                           pos=pos, arm=arm, shown=shown))
    print(f"[features] {len(frames)} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(frames=frames), f"{OUT}/frames_all.pt")

    # instruction check
    bases = sorted({f["base"] for f in frames})
    def stem_language(stem):  # 'LIVING_ROOM_SCENE2_put_both_...' -> 'put both ...' (as v10_key_film.language)
        parts = stem.split("_")
        i = next(i for i, p in enumerate(parts) if p.startswith("SCENE")) + 1
        return " ".join(parts[i:]).lower()

    base_lang = {b: stem_language(b) for b in bases}
    # the closed loop shows the base instruction outside the language category (eval_suite's fix_language)
    for f in frames:
        if f["cat"] != "Language Instructions":
            f["shown"] = base_lang[f["base"]]
    text = InstructionText.from_policy(policy, [base_lang[b] for b in bases])
    B = torch.stack([text.embed(base_lang[b]) for b in bases])
    inst = collections.defaultdict(list)
    for f in frames:
        s = B @ text.embed(f["shown"])
        order = [bases[i] for i in torch.argsort(-s).tolist()]
        rank = order.index(f["base"]) + 1
        inst[f["cat"]].append(rank)
        inst["all"].append(rank)

    # reference pick
    cams = {"agent": slice(0, 16), "wrist": slice(16, 32), "both": slice(0, 32)}
    unit = lambda v: torch.nn.functional.normalize(v.float().flatten(), dim=0)
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    by_task = collections.defaultdict(list)
    for f in frames:
        by_task[f["base"]].append(f)
    for base, fs in by_task.items():
        V = {cam: torch.stack([unit(f["vis"][sl]) for f in fs]) for cam, sl in cams.items()}
        Cw = torch.stack([unit(f["canon"][cams["wrist"]]) for f in fs])
        arms = np.stack([f["arm"] for f in fs])
        for i, q in enumerate(fs):
            cand = [j for j, c in enumerate(fs) if c["tid"] != q["tid"]]
            dist = {j: layout_dist(q["pos"], fs[j]["pos"]) for j in cand}
            same = [j for j in cand if dist[j] < 0.005]
            other = [j for j in cand if dist[j] >= 0.005]
            dq = np.linalg.norm(arms[cand] - q["arm"], axis=1)
            scores = {cam: (V[cam][cand] @ V[cam][i]).numpy() for cam in cams}
            qw = Cw[i] if q["cat"] == "Robot Initial States" else V["wrist"][i]
            scores["wrist-canon"] = (V["wrist"][cand] @ qw).numpy()
            w = np.exp(-dq / 0.05)
            scores["weighted"] = w * scores["wrist"] + (1 - w) * scores["agent"]
            for m, sc in scores.items():
                order = [cand[t] for t in np.argsort(-sc)]
                top = order[0]
                oth = [j for j in order if j in set(other)]
                for grp in ("all", q["cat"]):
                    r = res[m]
                    r[grp + "|has_same"].append(float(bool(same)))
                    if same:
                        r[grp + "|hit"].append(float(dist[top] < 0.005))
                    r[grp + "|d1"].append(dist[top])
                    if oth:
                        r[grp + "|d1_other"].append(dist[oth[0]])
                        r[grp + "|rand_other"].append(float(np.mean([dist[j] for j in oth])))
                        r[grp + "|best_other"].append(min(dist[j] for j in oth))
    out = {m: {k: float(np.mean(v)) for k, v in r.items()} for m, r in res.items()}
    out["instruction_rank"] = {g: dict(top1=float(np.mean([x == 1 for x in v])), top2=float(np.mean([x <= 2 for x in v])),
                                       top3=float(np.mean([x <= 3 for x in v])), n=len(v)) for g, v in inst.items()}
    json.dump(out, open(f"{OUT}/result_all.json", "w"), indent=1)
    print("\n[instruction] correct base task in top-1 / top-2 / top-3:")
    for g, v in out["instruction_rank"].items():
        print(f"  {g:24s} {100 * v['top1']:5.1f}% / {100 * v['top2']:5.1f}% / {100 * v['top3']:5.1f}%  (n={v['n']})")
    print("\n[pick] method: has-same | same-layout hit@1 | top-1 dist (cm) | without same: top-1 / random / best (cm)")
    for grp in ("all",) + K.CATS + NEW:
        print(f"-- queries: {grp}")
        for m, r in out.items():
            if m == "instruction_rank" or f"{grp}|d1" not in r:
                continue
            hit = f"{100 * r[grp + '|hit']:5.1f}%" if grp + "|hit" in r else "   -  "
            oth = (f"{100 * r[grp + '|d1_other']:5.2f} / {100 * r[grp + '|rand_other']:5.2f} / "
                   f"{100 * r[grp + '|best_other']:5.2f}") if grp + "|d1_other" in r else "-"
            print(f"  {m:12s} {100 * r[grp + '|has_same']:5.1f}% | {hit} | {100 * r[grp + '|d1']:5.2f} | {oth}")
    print(f"-> {OUT}/result_all.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
