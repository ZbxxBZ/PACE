"""
Scheme 1 of the phase redesign (10-30; user: "先做方案一的曲线"): which object governs the hand at each moment of a task,
read from the memory itself instead of gripper events. For every original libero_10 task, the LeRobot-era pi0.5
references (runs/v10b/refs, successes, init states 6..49) are replayed in their own scenes recording, per step, the hand
position, the gripper command and every body's position. The episodes of a task are aligned to one progress axis by
DTW (features: hand-to-object distances of the task's relevant bodies + gripper command), and at each progress k the
hand is expressed in each relevant object's frame (hand - object position, current positions; an object carried by the
hand is masked). The spread of that relative position across episodes (sigma, cm) says how consistently the hand sits
relative to that object; weight w_o(k) = (1 / sigma_o^2) / sum, with the world frame (absolute hand position) as a
baseline. Compared with the current gripper-event phases (v10_oracle_shift2.phase_targets): majority anchor per k, and
per episode a soft assignment p(o | episode at k) from the per-frame Gaussian likelihood, also on episodes whose gripper
command jitters (more open / close switches than the task's median + 2).
    python phase_curves.py --workers 8 --out runs/v11/phase_curves
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import pickle
import time

import numpy as np

import collect_pairs as P
import v10_oracle_shift2 as O

V = "/root/pace/runs/v10b"
CARRY_MOVE, CARRY_NEAR = 0.02, 0.10  # an object moved > 2 cm from its start and within 10 cm of the hand is carried


def replay(src):
    """Replay a reference in its own scene; per-step hand, gripper command, body positions; the rec of record_source."""
    env = O.make_env(src["task_id"], src["init_index"])
    try:
        env.reset(seed=int(src["seed"]))
        sim = O.Sim(env)
        acts = src["actions"]
        names = list(sim.bodies)
        hand, pos, grip = [], [], []
        start = sim.positions()
        events, closed, success, end = [], acts[0][6] > 0, False, len(acts)
        for t, a in enumerate(acts):
            p = sim.positions()
            hand.append(sim.hand())
            pos.append(np.stack([p[n] for n in names]))
            grip.append(float(a[6] > 0))
            now = a[6] > 0
            if now != closed and all((x[6] > 0) == now for x in acts[t:t + 3]):
                events.append(dict(t=t, kind="grasp" if now else "release", hand=sim.hand(), pos=p))
                closed = now
            _, _, _, _, info = env.step(a)
            if info.get("is_success"):
                success, end = True, t + 1
                break
        final = dict(t=end, hand=sim.hand(), pos=sim.positions())
        rec = dict(success=success, end=end, events=events, final=final, start=start, movable=sim.movable,
                   goal=[list(g) for g in sim.inner.parsed_problem["goal_state"]])
        switches = int(np.sum(np.abs(np.diff(grip)) > 0))
        return dict(key=src["key"], task=src["task_id"], names=names, hand=np.stack(hand), pos=np.stack(pos),
                    grip=np.array(grip), rec=rec, switches=switches)
    finally:
        env.close()


def dtw_path(a, b):
    """DTW between feature sequences a [n, f], b [m, f]; returns for each row of b the matched rows of a."""
    n, m = len(a), len(b)
    c = np.linalg.norm(a[:, None] - b[None], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = c[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, match = n, m, collections.defaultdict(list)
    while i > 0 and j > 0:
        match[j - 1].append(i - 1)
        k = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return [sorted(match[j]) for j in range(m)]


def analyse(eps):
    """One task: relevant bodies, DTW alignment to the medoid-length episode, sigma / weights per progress step."""
    names = eps[0]["names"]
    rel = set()
    for e in eps:
        for g in e["rec"]["goal"]:
            for arg in g[1:]:
                b = O.anchor_body(arg, names)
                if b in names:
                    rel.add(b)
        for ph in O.phase_targets(e["rec"]):
            if ph[2] in names:
                rel.add(ph[2])
    rel = sorted(rel)
    idx = [names.index(b) for b in rel]

    def feats(e):
        d = np.linalg.norm(e["hand"][:, None, :] - e["pos"][:, idx, :], axis=2)
        return np.concatenate([d / 0.1, e["grip"][:, None]], 1)

    lens = [len(e["hand"]) for e in eps]
    ref = eps[int(np.argsort(lens)[len(lens) // 2])]
    K = len(ref["hand"])
    fr = feats(ref)
    frames = rel + ["world"]
    relpos = {f: [[] for _ in range(K)] for f in frames}  # per progress k: list of (episode, rel vector)
    anchors = [collections.Counter() for _ in range(K)]
    per_ep = []
    for ei, e in enumerate(eps):
        match = dtw_path(feats(e), fr)
        phase_of = np.empty(len(e["hand"]), dtype=object)
        for (s, t_end, name) in O.phase_targets(e["rec"]):
            phase_of[s:max(t_end, s + 1)] = name
        phase_of[phase_of == None] = phase_of[[i for i in range(len(phase_of)) if phase_of[i] is not None][-1]]  # noqa: E711
        steps = []
        for k in range(K):
            t = match[k][len(match[k]) // 2]
            steps.append(t)
            h = e["hand"][t]
            for b, j in zip(rel, idx):
                p, p0 = e["pos"][t, j], e["pos"][0, j]
                carried = np.linalg.norm(p - p0) > CARRY_MOVE and np.linalg.norm(p - h) < CARRY_NEAR
                if not carried:
                    relpos[b][k].append((ei, h - p))
            relpos["world"][k].append((ei, h.copy()))
            anchors[k][phase_of[t]] += 1
        per_ep.append(dict(steps=np.array(steps), phase=[phase_of[t] for t in steps], switches=e["switches"]))
    n = len(eps)
    sig = {f: np.full(K, np.inf) for f in frames}
    mu = {f: np.zeros((K, 3)) for f in frames}
    for f in frames:
        for k in range(K):
            v = np.array([x for _, x in relpos[f][k]])
            if len(v) >= 0.6 * n:
                mu[f][k] = v.mean(0)
                sig[f][k] = 100 * np.sqrt(v.var(0).mean())  # cm, mean over axes
    prec = np.stack([1 / np.maximum(sig[f], 0.3) ** 2 for f in frames])  # floor 0.3 cm
    w = prec / prec.sum(0, keepdims=True)
    maj = [a.most_common(1)[0][0] for a in anchors]
    # per-episode soft assignment from the per-frame isotropic Gaussian likelihood (world excluded: objects only)
    agree_ep, flips = [], []
    for ei, pe in enumerate(per_ep):
        e = eps[ei]
        post_arg = []
        for k in range(K):
            t = pe["steps"][k]
            h = e["hand"][t]
            ll = []
            for b, j in zip(rel, idx):
                s = sig[b][k] / 100
                p, p0 = e["pos"][t, j], e["pos"][0, j]
                carried = np.linalg.norm(p - p0) > CARRY_MOVE and np.linalg.norm(p - h) < CARRY_NEAR
                if not np.isfinite(s) or carried:
                    ll.append(-np.inf)
                    continue
                r = (h - p) - mu[b][k]
                ll.append(-0.5 * (r @ r) / s ** 2 - 3 * np.log(s))
            post_arg.append(rel[int(np.argmax(ll))] if np.isfinite(max(ll)) else None)
        ok = [a == ph for a, ph in zip(post_arg, pe["phase"]) if a is not None and ph in rel]
        agree_ep.append(float(np.mean(ok)) if ok else np.nan)
        flips.append((sum(a != b for a, b in zip(post_arg[1:], post_arg[:-1])),
                      sum(a != b for a, b in zip(pe["phase"][1:], pe["phase"][:-1]))))
    obj_arg = [rel[int(np.argmax([w[frames.index(b)][k] for b in rel]))] for k in range(K)]
    agree_maj = float(np.mean([a == m for a, m in zip(obj_arg, maj) if m in rel]))
    return dict(rel=rel, frames=frames, sig=sig, w=w, maj=maj, K=K, ref_key=ref["key"], agree_maj=agree_maj,
                agree_ep=np.array(agree_ep), flips=np.array(flips), switches=np.array([pe["switches"] for pe in per_ep]),
                grip_ref=ref["grip"], n=n)


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = sorted(res)
    fig, axes = plt.subplots(len(tasks), 2, figsize=(15, 3.0 * len(tasks)), squeeze=False)
    for row, tid in enumerate(tasks):
        r = res[tid]
        x = np.linspace(0, 1, r["K"])
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax = axes[row, 0]
        for i, f in enumerate(r["frames"]):
            s = np.where(np.isfinite(r["sig"][f]), r["sig"][f], np.nan)
            ax.plot(x, s, label=f, color="0.5" if f == "world" else cols[i % len(cols)],
                    ls="--" if f == "world" else "-", lw=1.4)
        ax.set_yscale("log")
        ax.set_ylabel("spread across episodes (cm)")
        ax.set_title(f"{r['name']} (n={r['n']})", fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right")
        ax2 = axes[row, 1]
        ax2.stackplot(x, r["w"], labels=r["frames"],
                      colors=[("0.8" if f == "world" else cols[i % len(cols)]) for i, f in enumerate(r["frames"])])
        maj = r["maj"]
        for k in range(1, r["K"]):
            if maj[k] != maj[k - 1]:
                ax2.axvline(x[k], color="k", lw=1.2, ls=":")
        ax2.plot(x, 0.02 + 0.96 * r["grip_ref"], color="k", lw=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("weight")
        ax2.set_title(f"weights; dotted = gripper-phase boundaries; line = gripper (ref ep.); "
                      f"argmax = phase anchor {100 * r['agree_maj']:.0f}%", fontsize=8, loc="left")
    for ax in axes[-1]:
        ax.set_xlabel("task progress (DTW-aligned)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="/root/pace/runs/v11/phase_curves")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = f"{args.out}/replays.pkl"
    rows = [r for r in map(json.loads, open(f"{V}/refs/episodes.jsonl")) if r["success"]]
    if os.path.exists(cache):
        eps = pickle.load(open(cache, "rb"))
    else:
        srcs = [dict(key=f"t{r['task_id']}_i{r['init']}", task_id=int(r["task_id"]), init_index=int(r["init"]),
                     seed=int(r["seed"]), actions=np.load(f"{V}/refs/{r['actions']}").astype(np.float64)) for r in rows]
        t0, eps = time.time(), []
        with mp.get_context("spawn").Pool(args.workers, initializer=P.worker_init) as pool:  # original tasks: ids 10000+
            for i, e in enumerate(pool.imap_unordered(replay, srcs)):
                eps.append(e)
                if i % 40 == 0:
                    print(f"[replay] {i + 1}/{len(srcs)} ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump(eps, open(cache, "wb"))
    eps = [e for e in eps if e["rec"]["success"]]
    names = {r["task_id"]: r["language"] for r in rows}
    by = collections.defaultdict(list)
    for e in eps:
        by[e["task"]].append(e)
    res, lines = {}, []
    for tid in sorted(by):
        t0 = time.time()
        r = analyse(by[tid])
        r["name"] = names[tid]
        res[tid] = r
        jit = r["switches"] > np.median(r["switches"]) + 2
        line = (f"{tid} {names[tid][:60]:60s} n={r['n']:2d} rel={r['rel']} | argmax==phase anchor {100 * r['agree_maj']:.0f}% "
                f"| per-episode agreement median {100 * np.nanmedian(r['agree_ep']):.0f}% "
                f"| jittery eps {int(jit.sum())}: agreement {100 * np.nanmedian(r['agree_ep'][jit]) if jit.any() else float('nan'):.0f}%, "
                f"switches soft/phase {r['flips'][jit].mean(0).round(1).tolist() if jit.any() else '-'} "
                f"(others {r['flips'][~jit].mean(0).round(1).tolist()}) ({time.time() - t0:.0f}s)")
        print(line, flush=True)
        lines.append(line)
    pickle.dump({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, open(f"{args.out}/curves.pkl", "wb"))
    open(f"{args.out}/summary.txt", "w").write("\n".join(lines) + "\n")
    plot(res, f"{args.out}/phase_curves.png")
    print(f"-> {args.out}/phase_curves.png", flush=True)


if __name__ == "__main__":
    main()
