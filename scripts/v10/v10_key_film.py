"""
V10 reference-pick offline check, part 2 (10-23): raw cosine vs task-centered PCA vs a learned key, with and without
FiLM language modulation, on the call-0 image tokens written by v10_key_pca.py (runs/rectest/v10_keypca/frames.pt:
10 base tasks x 10 init states x 8 visual variants).
  raw      : cosine of the flattened unit cells (agentview / wrist / both);
  centered : minus the task's mean frame;  pca-k : centered, top-k components (fit on the task's frames, query
             included: optimistic for PCA);
  learned  : per-cell linear map 2048 -> 64 shared over the 32 cells (keeps where things are), unit-normalized
             flattened key; trained so that the softmax over a frame's same-task candidates (other variants) matches
             exp(-layout distance / 1 cm): same layout first, then the nearest layouts;
  film     : the same, with the cells' 64 channels scaled and shifted by gamma, beta = MLP(instruction embedding)
             (pi0.5's own IDF-weighted token embedding, pace.pi05.InstructionText) before normalizing.
The learned keys are cross-validated over tasks: 5 folds, trained on 8 base tasks, tested on the other 2 (unseen
tasks and unseen instructions), 3 seeds each. Same metrics for every method, every task tested exactly once:
same-layout hit rate of the top-1 (chance 10%) and the top-1's layout distance; with the same-layout candidates
removed, the top-1's distance against random and the best possible.
    python v10_key_film.py -> runs/rectest/v10_keypca/result_film.json
"""
import collections
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIR = "/root/pace/runs/rectest/v10_keypca"
CATS = ("Light Conditions", "Sensor Noise", "Background Textures", "Camera Viewpoints")


def language(stem):
    """'LIVING_ROOM_SCENE2_put_both_...' -> 'put both ...'."""
    parts = stem.split("_")
    i = next(i for i, p in enumerate(parts) if p.startswith("SCENE")) + 1
    return " ".join(parts[i:]).lower()


def layout_dist(layouts):
    """base -> init -> init -> largest xy displacement of a movable object (m)."""
    out = {}
    for b, L in layouts.items():
        L = {int(k): {n: np.array(v) for n, v in d.items()} for k, d in L.items()}
        out[b] = {a: {c: max(float(np.linalg.norm(L[a][n] - L[c][n])) for n in L[a]) for c in L} for a in L}
    return out


def metrics(fs, sim, D, rec):
    """Top-1 metrics of every query over its same-task candidates of the other variants, into rec[group]."""
    for i, q in enumerate(fs):
        cand = [j for j, c in enumerate(fs) if c["tid"] != q["tid"]]
        order = sorted(cand, key=lambda j: -float(sim[i, j]))
        top = fs[order[0]]
        other = [j for j in order if fs[j]["init"] != q["init"]]
        row = dict(hit=float(top["init"] == q["init"]), d1=D[q["init"]][top["init"]],
                   d1_other=D[q["init"]][fs[other[0]]["init"]],
                   rand_other=float(np.mean([D[q["init"]][fs[j]["init"]] for j in other])),
                   best_other=min(D[q["init"]][fs[j]["init"]] for j in other))
        for g in ("all", q["cat"]):
            for k, v in row.items():
                rec[g][k].append(v)


class Key(nn.Module):
    def __init__(self, d_in=2048, d=64, film=False, d_lang=2048):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.film = nn.Sequential(nn.Linear(d_lang, 128), nn.GELU(), nn.Linear(128, 2 * d)) if film else None
        if film:
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)  # starts as the no-language key

    def forward(self, vis, lang):  # vis [N, 32, D], lang [N, Dl]
        h = self.proj(vis)
        if self.film is not None:
            g, b = self.film(lang).chunk(2, dim=-1)
            h = (1 + g[:, None]) * h + b[:, None]
        return F.normalize(h.flatten(1), dim=-1)


def train_key(train_tasks, by_task, D, lang, film, seed, steps=400, tau=0.07, tau_d=0.01):
    torch.manual_seed(seed)
    m = Key(film=film)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    data = []
    for b in train_tasks:
        fs = by_task[b]
        X = torch.stack([f["vis"].float() for f in fs])
        Lg = lang[b].expand(len(fs), -1)
        dist = torch.tensor([[D[b][f["init"]][g["init"]] for g in fs] for f in fs])
        same_var = torch.tensor([[f["tid"] == g["tid"] for g in fs] for f in fs])
        data.append((X, Lg, dist, same_var))
    for step in range(steps):
        loss = 0.0
        for X, Lg, dist, same_var in data:
            k = m(X, Lg)
            logits = (k @ k.T) / tau
            logits = logits.masked_fill(same_var, -1e9)  # candidates: the other variants only
            target = torch.softmax((-dist / tau_d).masked_fill(same_var, -1e9), dim=1)
            loss = loss + -(target * torch.log_softmax(logits, dim=1)).sum(1).mean()
        opt.zero_grad()
        (loss / len(data)).backward()
        opt.step()
    return m


def main():
    blob = torch.load(f"{DIR}/frames.pt", weights_only=False)
    frames, D = blob["frames"], layout_dist(blob["layouts"])
    by_task = collections.defaultdict(list)
    for f in frames:
        by_task[f["base"]].append(f)
    tasks = sorted(by_task)
    from pace.pi05 import InstructionText

    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=[language(b) for b in tasks])
    lang = {b: text.embed(language(b))[None].float() for b in tasks}
    res = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    cams = {"agent": slice(0, 16), "wrist": slice(16, 32), "both": slice(0, 32)}
    for b in tasks:  # unsupervised methods, per task
        fs = by_task[b]
        for cam, sl in cams.items():
            X = torch.stack([f["vis"][sl].float().flatten() for f in fs]).double()
            Xc = X - X.mean(0, keepdim=True)
            reps = {"raw": X, "centered": Xc}
            _, _, V = torch.linalg.svd(Xc, full_matrices=False)
            for k in (8, 16, 32, 64):
                reps[f"pca{k}"] = Xc @ V[:k].T
            for name, R in reps.items():
                Rn = F.normalize(R, dim=1)
                metrics(fs, Rn @ Rn.T, D[b], res[f"{cam}/{name}"])
    rng = random.Random(0)
    order = tasks[:]
    rng.shuffle(order)
    folds = [order[i::5] for i in range(5)]
    for film in (False, True):
        name = "both/film" if film else "both/learned"
        for seed in range(3):
            for test in folds:
                train = [b for b in tasks if b not in test]
                m = train_key(train, by_task, D, lang, film, seed)
                with torch.no_grad():
                    for b in test:
                        fs = by_task[b]
                        k = m(torch.stack([f["vis"].float() for f in fs]), lang[b].expand(len(fs), -1))
                        metrics(fs, k @ k.T, D[b], res[name])
            print(f"[{name}] seed {seed} done", flush=True)
    out = {m: {g: {k: float(np.mean(v)) for k, v in r.items()} for g, r in rr.items()} for m, rr in res.items()}
    json.dump(out, open(f"{DIR}/result_film.json", "w"), indent=1)
    print("\nmethod: same-layout hit@1 | top-1 dist (cm) | without same-layout: top-1 / random / best (cm)")
    for g in ("all",) + CATS:
        print(f"-- queries: {g}")
        for m, rr in out.items():
            r = rr[g]
            print(f"  {m:16s} {100 * r['hit']:5.1f}% | {100 * r['d1']:5.2f} | {100 * r['d1_other']:5.2f} / "
                  f"{100 * r['rand_other']:5.2f} / {100 * r['best_other']:5.2f}")


if __name__ == "__main__":
    main()
