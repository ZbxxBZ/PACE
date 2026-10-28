"""
V10 reference-pick offline check (10-23): does a task-centered PCA of the call-0 image tokens pick the reference whose
object layout is closest better than the raw cosine does, when the appearance varies?
Frames: the 10 libero_10 base tasks x init states 0..9 (10 layouts, objects 2-4 cm apart) x 8 LIBERO-plus variants
(2 each of light / sensor noise / background / camera; these share the default layout, so a (task, init) has the same
physical scene in all of them). Each frame is the env's reset observation, through the closed loop's own preprocessing
(preprocess_observation -> task -> env preprocessor -> policy preprocessor) and pi0.5's frozen image embedding:
PACE's image tokens (16 unit cells per camera, agentview then wrist; pace.pi05.call_keys). CPU only (OSMesa, torch).
Layout distance of two frames: the largest xy displacement of a movable object between their init states (simulator).
For every query frame the candidates are the task's frames of the other 7 variants; ranked by
  raw      : cosine of the flattened cells (= mean cell cosine), agentview / wrist / both;
  centered : minus the task's mean frame, cosine;
  pca-k    : centered, top-k principal components (fit on the task's 80 frames, query included: optimistic for PCA).
Metrics: same-layout hit rate of the top-1 (chance 10%), mean layout distance of the top-1; and with the same-layout
candidates removed, the top-1's distance against random and the best possible.
    python v10_key_pca.py --workers 24 -> runs/rectest/v10_keypca/{frames.pt,result.json}
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

OUT = "/root/pace/runs/rectest/v10_keypca"
CATS = ("Light Conditions", "Sensor Noise", "Background Textures", "Camera Viewpoints")
G = {}


def worker_init():
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.libero import _get_suite, _make_env_fns, parse_camera_names

    cfg = LiberoPlusEnv(task="libero_10")
    gk = dict(cfg.gym_kwargs)
    gk.pop("task_ids", None)
    G.update(cfg=cfg, suite=_get_suite("libero_10"), gk=gk, make_fns=_make_env_fns,
             cams=parse_camera_names(cfg.camera_name))


def render(job):
    """(tid, init) -> the preprocessed reset observation (before the env preprocessor) and the bodies' positions."""
    import gymnasium as gym
    from lerobot.envs.utils import preprocess_observation

    tid, k, seed = job
    c = G["cfg"]
    fn = G["make_fns"](suite=G["suite"], suite_name="libero_10", task_id=int(tid), n_envs=1, camera_names=G["cams"],
                       episode_length=c.episode_length, init_states=c.init_states, gym_kwargs=G["gk"],
                       control_mode=c.control_mode, camera_name_mapping=c.camera_name_mapping,
                       is_libero_plus=c.is_libero_plus)[0]
    env = gym.vector.SyncVectorEnv([functools.partial(fn.func, int(k), **fn.keywords)])
    try:
        obs, _ = env.reset(seed=[int(seed)])
        inner = env.envs[0]._env.env
        pos = {n: np.array(inner.sim.data.body_xpos[i], dtype=np.float64) for n, i in inner.obj_body_id.items()}
        o = preprocess_observation(obs)
        o["task"] = list(env.call("task_description"))
        # numpy across the process boundary: torch tensors go through shared-memory fds, which run out at 800 frames
        return tid, k, to_numpy(o), pos, sorted(inner.objects_dict)
    finally:
        env.close()


def to_numpy(x):
    """Nested dicts / lists of tensors -> numpy (robot_state is a dict of tensors)."""
    if torch.is_tensor(x):
        return x.numpy()
    if isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)
    return x


def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_torch(v) for v in x)
    return x


def pick_variants(per_cat=2, seed=0):
    """Per base task, `per_cat` variants of each visual category, from all 2519 (base = the original bddl stem)."""
    from pace.eval_suite import classification
    from lerobot.envs.libero import _get_suite
    import libero.libero as L

    suite = _get_suite("libero_10")
    cls = classification("libero_10")
    # the 10 base tasks: the default layout's init files (every visual variant of a task uses its .pruned_init)
    stems = sorted({f[:-len(".pruned_init")] for f in os.listdir(os.path.join(os.path.dirname(L.__file__), "init_files",
                                                                              "libero_10")) if f.endswith(".pruned_init")})
    assert len(stems) == 10, stems
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for tid in range(suite.n_tasks):
        t = suite.get_task(tid)
        cat = cls.get(t.name, (None,))[0]
        if cat not in CATS:
            continue
        base = max((s for s in stems if t.name.startswith(s)), key=len, default=None)
        if base:
            by[base][cat].append(tid)
    rng = random.Random(seed)
    out = {}
    for base, d in sorted(by.items()):
        if all(len(d[c]) >= per_cat for c in CATS):
            out[base] = {c: sorted(rng.sample(sorted(d[c]), per_cat)) for c in CATS}
    return out


def analyse(frames, ks=(8, 16, 32, 64)):
    """frames: list of dicts(task, cat, tid, init, vis [32, D], disp table) -> metrics per method."""
    by_task = collections.defaultdict(list)
    for f in frames:
        by_task[f["base"]].append(f)
    methods = collections.OrderedDict()
    cams = {"agent": slice(0, 16), "wrist": slice(16, 32), "both": slice(0, 32)}
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for base, fs in by_task.items():
        D = fs[0]["dist"]  # init -> init -> layout distance (m)
        for cam, sl in cams.items():
            X = torch.stack([f["vis"][sl].float().flatten() for f in fs]).double()
            reps = {"raw": X}
            Xc = X - X.mean(0, keepdim=True)
            reps["centered"] = Xc
            U, S, V = torch.linalg.svd(Xc, full_matrices=False)
            for k in ks:
                if k < len(fs):
                    reps[f"pca{k}"] = Xc @ V[:k].T
            for name, R in reps.items():
                Rn = torch.nn.functional.normalize(R, dim=1)
                sim = Rn @ Rn.T
                for i, q in enumerate(fs):
                    cand = [j for j, c in enumerate(fs) if c["tid"] != q["tid"]]
                    order = sorted(cand, key=lambda j: -float(sim[i, j]))
                    top = fs[order[0]]
                    dist = D[q["init"]][top["init"]]
                    other = [j for j in order if fs[j]["init"] != q["init"]]
                    d_other = D[q["init"]][fs[other[0]]["init"]]
                    rand_other = float(np.mean([D[q["init"]][fs[j]["init"]] for j in other]))
                    best_other = min(D[q["init"]][fs[j]["init"]] for j in other)
                    for grp in ("all", q["cat"]):
                        r = res[(cam, name)]
                        r[grp + "|hit"].append(float(top["init"] == q["init"]))
                        r[grp + "|d1"].append(dist)
                        r[grp + "|d1_other"].append(d_other)
                        r[grp + "|rand_other"].append(rand_other)
                        r[grp + "|best_other"].append(best_other)
    out = {}
    for (cam, name), r in res.items():
        out[f"{cam}/{name}"] = {k: float(np.mean(v)) for k, v in r.items()}
    return out


def render_all(jobs, workers, rounds=5):
    """Render every job in a spawn process pool; a worker that dies breaks the pool (no silent hang): the jobs it had
    not returned are resubmitted to a fresh pool, up to `rounds` times."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from concurrent.futures.process import BrokenProcessPool

    done, todo = {}, list(jobs)
    for rnd in range(rounds):
        if not todo:
            break
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=worker_init) as ex:
            futs = {ex.submit(render, j): j for j in todo}
            try:
                for f in as_completed(futs):
                    r = f.result()
                    done[(r[0], r[1])] = r
                    if len(done) % 50 == 0:
                        print(f"[render] {len(done)}/{len(jobs)}", flush=True)
            except BrokenProcessPool as e:
                print(f"[render] pool broke in round {rnd} ({e}); {len(done)}/{len(jobs)} done", flush=True)
        todo = [j for j in jobs if (j[0], j[1]) not in done]
    assert not todo, f"{len(todo)} jobs never rendered"
    return list(done.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--per_cat", type=int, default=2)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    picked = pick_variants(args.per_cat)
    print(f"[variants] {len(picked)} base tasks x {sum(len(v) for v in next(iter(picked.values())).values())} variants",
          flush=True)
    meta = {tid: (base, cat) for base, d in picked.items() for cat, tids in d.items() for tid in tids}
    jobs = [(tid, k, 1000 + k) for tid in sorted(meta) for k in args.inits]
    torch.multiprocessing.set_sharing_strategy("file_system")
    rendered = render_all(jobs, args.workers)
    print(f"[render] {len(rendered)} frames ({time.time() - t0:.0f}s)", flush=True)

    torch.set_num_threads(args.threads)
    from lerobot.envs.configs import LiberoPlusEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from pace.common import load_policy
    from pace.pi05 import call_keys

    policy, preprocessor = load_policy("lerobot/pi05_libero_finetuned", args.device, n_action_steps=10)
    env_pre, _ = make_env_pre_post_processors(LiberoPlusEnv(task="libero_10"), policy.config)
    frames, layouts = [], collections.defaultdict(dict)
    for tid, k, o, pos, movable in sorted(rendered, key=lambda x: (x[0], x[1])):
        o = to_torch(o)
        batch = preprocessor(env_pre(o))
        _, _, vis, _ = call_keys(policy, batch, cameras=(0, 1), sig_grid=2, features=True)
        base, cat = meta[tid]
        mv = {n: pos[n][:2] for n in movable if n in pos}
        if k in layouts[base]:  # the same (task, init) is one physical scene in every visual variant
            ref = layouts[base][k]
            dev = max(float(np.linalg.norm(mv[n] - ref[n])) for n in ref)
            assert dev < 1e-6, f"{base} init {k}: variant {tid} differs from the layout by {dev}"
        else:
            layouts[base][k] = mv
        frames.append(dict(base=base, cat=cat, tid=tid, init=k, vis=vis[0].half().cpu()))
    print(f"[features] {len(frames)} ({time.time() - t0:.0f}s)", flush=True)
    for f in frames:
        L_ = layouts[f["base"]]
        f["dist"] = {a: {b: max(float(np.linalg.norm(L_[a][n] - L_[b][n])) for n in L_[a]) for b in L_} for a in L_}
    torch.save(dict(frames=[{k: v for k, v in f.items() if k != "dist"} for f in frames],
                    layouts={b: {k: {n: v.tolist() for n, v in d.items()} for k, d in L_.items()}
                             for b, L_ in layouts.items()}), f"{OUT}/frames.pt")
    res = analyse(frames)
    json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
    print("\n[result] method: same-layout hit@1 | top-1 dist (cm) | without same-layout: top-1 / random / best (cm)")
    for grp in ("all",) + CATS:
        print(f"-- queries: {grp}")
        for m, r in res.items():
            if f"{grp}|hit" in r:
                print(f"  {m:18s} {100 * r[grp + '|hit']:5.1f}% | {100 * r[grp + '|d1']:5.2f} | "
                      f"{100 * r[grp + '|d1_other']:5.2f} / {100 * r[grp + '|rand_other']:5.2f} / "
                      f"{100 * r[grp + '|best_other']:5.2f}")
    print(f"-> {OUT}/result.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
