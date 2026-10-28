"""
V10-1b: train the comparison module (10-24; docs/PACE_V1-V8.md "V10-1b 结构与数据采集方案", "d 能从画面估多准",
"V10-1b 训练数据采集结果与 α 的定法"). pi0.5 stays frozen; its SigLIP features of every frame are precomputed
(extract_feats.py, PCA-384).
One sample = one policy call t of a pair episode (current scene following a reference):
  current   : both cameras at t (agentview dropped with prob. --drop_agent, so d leans on the wrist camera, whose pose is
              known; the camera-perturbed agentview cannot give metric positions -- localization probe 10-24);
  reference : both cameras at the aligned steps t-10, t, t+10, t+20 (the expert follows the reference's timeline, so
              the aligned step is t) and at the end of the current phase (the grasp / release frame);
  state     : hand offset from the reference at t (current eef - reference eef), both grippers, phase progress;
  language  : pi0.5's instruction vectors (InstructionText) of the current task and of the reference's task.
Outputs: d (xy shift of the reference path for this phase, m), and two logits: "following this reference succeeds"
(label: the expert pair's replay outcome; negatives 0; perturbed episodes masked) and "pi0.5 alone succeeds here"
(label: pi0.5 under the same scene and augmentation, collect_pi05.py --set orig_aug). The closed loop follows only
when the first exceeds the second by a margin set on validation (user's choice 10-24).
d labels: the true d (BDDL anchors) for expert and perturbed episodes (perturbed: the hand was pushed off by 1-3 cm,
the label stays the true d, so the offset input alone does not give the answer); negatives masked.
Validation (init states 45-49, with their own references): d error median / 75th / 90th percentile by the reference
hand's distance to the phase's contact point, on perturbed and expert episodes separately (registered gate: < 3 cm
median <= 1 cm, 10-20 cm median <= 2 cm on perturbed), AUROC of both heads.
    python train_v10b.py --out runs/v10b/train/run1
"""
import argparse
import collections
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V = "/root/pace/runs/v10b"
CALL = 10
SLOTS = 6  # current, ref t-10, t, t+10, t+20, ref phase end


# ----------------------------------------------------------------------------------------------------------------- data
def frame_meta(path):
    z = np.load(path)
    return dict(t=z["t"], eef=z["eef_pos"].astype(np.float32), grip=z["gripper"].astype(np.float32),
                success=bool(z["success"]))


def pair_rank(rows):
    rank, seen = {}, {}
    for r in sorted(rows, key=lambda r: (r["task_id"], r["init"], r["ref_task"], r["ref_init"])):
        sc = (r["task_id"], r["init"])
        rank[r["key"]] = seen.get(sc, 0)
        seen[sc] = rank[r["key"]] + 1
    return rank


def build_index(args):
    """Every (episode, call) sample with its labels; references' frame metadata. Cached in <out>/../index.pt."""
    cache = (f"{os.path.dirname(args.out.rstrip('/'))}/index_{args.feats.rstrip('/').split('/')[-1]}_{args.dlabel}"
             f"{'' if args.kinds == ['expert', 'perturbed', 'negative'] else '_' + '-'.join(args.kinds)}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)
    t0 = time.time()
    refs_rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
    lang = {r["task_id"]: r["language"] for r in refs_rows}
    pi = {(r["task_id"], r["init"], r["aug_idx"]): r["success"] for r in map(json.loads, open(f"{V}/pi05_aug/episodes.jsonl"))}
    refs = {}
    for r in refs_rows:
        key = f"t{r['task_id']}_i{r['init']}"
        for a in range(2):
            if os.path.exists(f"{args.feats}/refs/{key}_a{a}.npy"):
                refs.setdefault(key, {})[a] = frame_meta(f"{V}/frames/refs/{key}_a{a}.npz")
    samples, eps = [], []
    for kind in args.kinds:
        rows = [json.loads(l) for l in open(f"{V}/pairs/{kind}.jsonl")]
        rank = pair_rank(rows)
        for r in rows:
            fkey = f"{r['key']}_a0"
            if not os.path.exists(f"{args.feats}/{kind}/{fkey}.npy"):
                continue
            rkey = f"t{r['ref_task']}_i{r['ref_init']}"
            if rkey not in refs:
                continue
            m = frame_meta(f"{V}/frames/{kind}/{fkey}.npz")
            z = np.load(f"{V}/pairs/{kind}/{r['key']}.npz")
            d, path, phases = z["d"], z["path"], z["phases"]
            n = len(d)
            kin = None
            if args.dlabel == "kin" and kind != "negative":  # kinematic grasp-point labels (kin_labels.py)
                kp = f"{V}/kin_labels/{r['key']}.npy"
                kin = np.load(kp) if os.path.exists(kp) else None
            pidx = rank[r["key"]] % 3
            base = pi.get((r["task_id"], r["init"], pidx))
            follow = None if kind in ("perturbed", "carry") else (m["success"] if kind == "expert" else False)
            e = len(eps)
            eps.append(dict(kind=kind, key=r["key"], fkey=fkey, ref=rkey, split=r["split"], task=r["task_id"],
                            ref_task=r["ref_task"], meta=m, follow=follow, base=base,
                            phases=phases.astype(np.int64), n=n))
            for c, t in enumerate(m["t"]):
                tt = min(int(t), n - 1)
                ph = next((i for i, (s, e_) in enumerate(phases) if tt < max(e_, s + 1)), len(phases) - 1)
                s, e_ = phases[ph]
                contact = path[min(max(e_ - 1, 0), n - 1), :2]
                cs = int(np.argmin(np.abs(m["t"] - s)))  # this episode's call nearest the phase start (--state phase)
                samples.append(dict(ep=e, c=c, t=int(t), phase=ph, end=int(e_), start=int(s), cs=cs,
                                    d=(None if kind == "negative" else d[tt, :2].astype(np.float32) if args.dlabel == "true"
                                       else kin[tt].astype(np.float32) if kin is not None else None),
                                    d_true=d[tt, :2].astype(np.float32) if kind != "negative" else None,
                                    dist=float(np.linalg.norm(path[tt, :2] - contact))))
    idx = dict(samples=samples, eps=eps, refs=refs, lang=lang)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(idx, cache)
    print(f"[index] {len(samples)} samples, {len(eps)} episodes, {len(refs)} references ({time.time() - t0:.0f}s)",
          flush=True)
    return idx


class PairData(torch.utils.data.Dataset):
    def __init__(self, idx, split, args, lang_vec, train):
        self.idx, self.args, self.train, self.lang_vec = idx, args, train, lang_vec
        self.items = [i for i, s in enumerate(idx["samples"]) if idx["eps"][s["ep"]]["split"] == split]

    def __len__(self):
        return len(self.items)

    def _feat(self, kind, fkey, rows):
        a = np.load(f"{self.args.feats}/{kind}/{fkey}.npy", mmap_mode="r")
        return torch.from_numpy(np.ascontiguousarray(a[rows])).float()  # [k, 2, 256, D]

    def __getitem__(self, i):
        s = self.idx["samples"][self.items[i]]
        ep = self.idx["eps"][s["ep"]]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.items[i])
        refs = self.idx["refs"][ep["ref"]]
        a = int(rng.choice(sorted(refs))) if self.train else min(refs)
        rm = refs[a]
        rt = rm["t"]
        t = s["t"]
        want = [t - CALL, t, t + CALL, t + 2 * CALL, s["end"]]
        rows = [int(np.argmin(np.abs(rt - w))) for w in want]
        cur = self._feat(ep["kind"], ep["fkey"], [s["c"]])
        ref = self._feat("refs", f"{ep['ref']}_a{a}", rows)
        feats = torch.cat([cur, ref])  # [6, 2, 256, D]
        m = ep["meta"]
        j = rows[1]
        prog = (t - s["start"]) / max(s["end"] - s["start"], 1)
        off = m["eef"][s["c"]] - rm["eef"][j]
        if self.args.state == "phase":  # relative to the offset at this phase's start: no carried-over shift (V10-3)
            js = int(np.argmin(np.abs(rt - s["start"])))
            off = off - (m["eef"][s.get("cs", s["c"])] - rm["eef"][js])
        state = np.r_[off, m["grip"][s["c"]], rm["grip"][j], min(prog, 1.5), t / 600.0]
        drop = self.train and rng.random() < self.args.drop_agent
        lab_d = torch.tensor(s["d"]) if s["d"] is not None else torch.zeros(2)
        dt = s.get("d_true", s["d"])
        return dict(feats=feats, state=torch.tensor(state, dtype=torch.float32), drop=torch.tensor(drop),
                    lang=torch.stack([self.lang_vec[ep["task"]], self.lang_vec[ep["ref_task"]]]),
                    d=lab_d, d_ok=torch.tensor(s["d"] is not None),
                    d_true=torch.tensor(dt) if dt is not None else torch.zeros(2), dt_ok=torch.tensor(dt is not None),
                    follow=torch.tensor(float(ep["follow"] or 0)), follow_ok=torch.tensor(ep["follow"] is not None),
                    base=torch.tensor(float(ep["base"] or 0)), base_ok=torch.tensor(ep["base"] is not None),
                    dist=torch.tensor(s["dist"]), kind=["expert", "perturbed", "negative", "carry"].index(ep["kind"]), phase=torch.tensor(s["phase"]))


# ---------------------------------------------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, w, heads, cross=True):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(w), nn.MultiheadAttention(w, heads, batch_first=True)
        self.cross = cross
        if cross:
            self.n2, self.ca, self.nk = nn.LayerNorm(w), nn.MultiheadAttention(w, heads, batch_first=True), nn.LayerNorm(w)
        self.n3, self.mlp = nn.LayerNorm(w), nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))

    def forward(self, x, kv=None, q_mask=None, kv_mask=None):
        h = self.n1(x)
        x = x + self.sa(h, h, h, key_padding_mask=q_mask, need_weights=False)[0]
        if self.cross:
            k = self.nk(kv)
            x = x + self.ca(self.n2(x), k, k, key_padding_mask=kv_mask, need_weights=False)[0]
        return x + self.mlp(self.n3(x))


class Comparator(nn.Module):
    def __init__(self, dim_in, w=256, heads=8, layers=2, qlayers=2, nq=8, lang_dim=2048, state_dim=9, pool=1):
        super().__init__()
        self.pool = pool
        g = 16 // pool
        self.proj = nn.Sequential(nn.Linear(dim_in, w), nn.LayerNorm(w))
        self.pos = nn.Parameter(torch.zeros(1, 1, 1, g * g, w))
        self.cam = nn.Parameter(torch.zeros(1, 1, 2, 1, w))
        self.slot = nn.Parameter(torch.zeros(1, SLOTS, 1, 1, w))
        for p in (self.pos, self.cam, self.slot):
            nn.init.normal_(p, std=0.02)
        self.ref_enc = nn.ModuleList([Block(w, heads, cross=False)])
        self.cmp = nn.ModuleList([Block(w, heads) for _ in range(layers)])
        self.queries = nn.Parameter(torch.randn(1, nq, w) * 0.02)
        self.state = nn.Sequential(nn.Linear(state_dim, w), nn.GELU(), nn.Linear(w, w))
        self.lang = nn.Linear(lang_dim, w)
        self.qf = nn.ModuleList([Block(w, heads) for _ in range(qlayers)])
        self.out = nn.LayerNorm(w)
        self.head_d = nn.Linear(nq * w, 2)
        self.head_follow = nn.Linear(nq * w, 1)
        self.head_base = nn.Linear(nq * w, 1)

    def forward(self, feats, state, lang, drop, d_scale, return_h=False):
        B = feats.shape[0]
        x = feats
        if self.pool > 1:  # [B, 6, 2, 256, D] -> pooled grid
            x = x.view(B * SLOTS * 2, 16, 16, -1).permute(0, 3, 1, 2)
            x = F.avg_pool2d(x, self.pool).permute(0, 2, 3, 1).reshape(B, SLOTS, 2, -1, x.shape[1])
        x = self.proj(x) + self.pos + self.cam + self.slot  # [B, 6, 2, T, w]
        T = x.shape[3]
        mask = torch.zeros(B, SLOTS, 2, T, dtype=torch.bool, device=x.device)
        mask[:, :, 0] = drop[:, None, None]  # agentview dropped (current and reference)
        cur, ref = x[:, 0].reshape(B, 2 * T, -1), x[:, 1:].reshape(B, (SLOTS - 1) * 2 * T, -1)
        cm, rm = mask[:, 0].reshape(B, -1), mask[:, 1:].reshape(B, -1)
        for blk in self.ref_enc:
            ref = blk(ref, q_mask=rm)
        for blk in self.cmp:
            cur = blk(cur, ref, q_mask=cm, kv_mask=rm)
        q = torch.cat([self.queries.expand(B, -1, -1), self.state(state)[:, None], self.lang(lang)], 1)
        for blk in self.qf:
            q = blk(q, cur, kv_mask=cm)
        h = self.out(q[:, :self.queries.shape[1]]).flatten(1)
        out = d_scale * torch.tanh(self.head_d(h)), self.head_follow(h)[:, 0], self.head_base(h)[:, 0]
        return out + (h,) if return_h else out  # h: the pooled queries, input of the V10-3 value head


# ------------------------------------------------------------------------------------------------------------- training
def auroc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


BINS = [(0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.20), (0.20, 1.0)]


@torch.no_grad()
def evaluate(model, loader, args):
    model.eval()
    rec = collections.defaultdict(list)
    for b in loader:
        b = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, lf, lb = model(b["feats"], b["state"], b["lang"], b["drop"], args.d_scale)
        err = (d.float() - b["d_true"]).norm(dim=1)
        for k, v in dict(err=err, dist=b["dist"], kind=b["kind"], d_ok=b["dt_ok"], lf=lf.float(), lb=lb.float(), phase=b["phase"],
                         follow=b["follow"], follow_ok=b["follow_ok"], base=b["base"], base_ok=b["base_ok"],
                         zero=b["d_true"].norm(dim=1), offset=(b["state"][:, :2] - b["d_true"]).norm(dim=1)).items():
            rec[k].append(v.cpu())
    r = {k: torch.cat(v).numpy() for k, v in rec.items()}
    out = {}
    for kind, nm in ((1, "perturbed"), (0, "expert")):
        for lo, hi in BINS:
            sel = (r["kind"] == kind) & r["d_ok"] & (r["dist"] >= lo) & (r["dist"] < hi)
            if sel.sum():
                e = 100 * r["err"][sel]
                # + medians of two trivial answers: d = 0, and d = the hand's offset from the reference (the state input)
                out[f"{nm}_{int(100 * lo)}-{int(100 * hi)}"] = [round(float(np.median(e)), 2),
                                                                round(float(np.percentile(e, 75)), 2),
                                                                round(float(np.percentile(e, 90)), 2), int(sel.sum()),
                                                                round(float(np.median(100 * r["zero"][sel])), 2),
                                                                round(float(np.median(100 * r["offset"][sel])), 2)]
    sel = r["kind"] == 1
    out["perturbed_all_median"] = round(float(np.median(100 * r["err"][sel & r["d_ok"]])), 3)
    for kind, nm in ((1, "perturbed"), (0, "expert"), (3, "carry")):  # first phase vs later phases (carryover)
        for lo_ph, nm_ph in ((0, "ph0"), (1, "ph1+")):
            sel = (r["kind"] == kind) & r["d_ok"] & ((r["phase"] == 0) if lo_ph == 0 else (r["phase"] >= 1))
            if sel.sum():
                out[f"{nm}_{nm_ph}_median"] = round(float(np.median(100 * r["err"][sel])), 3)
    fo, bo = r["follow_ok"].astype(bool), r["base_ok"].astype(bool)
    out["auroc_follow"] = round(auroc(r["follow"][fo] > 0.5, r["lf"][fo]), 3)
    out["auroc_base"] = round(auroc(r["base"][bo] > 0.5, r["lb"][bo]), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats", default=f"{V}/feats")
    ap.add_argument("--dlabel", choices=["true", "kin"], default="true",
                    help="d training label: simulator object displacement, or kinematic grasp-point hand difference "
                         "(kin_labels.py; validation is always scored against the true displacement)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--kinds", nargs="+", default=["expert", "perturbed", "negative"],
                    help="pair kinds to train on (+ carry: the hand enters each phase with the previous phase's d, V10-3)")
    ap.add_argument("--state", choices=["abs", "phase"], default="abs",
                    help="hand-offset input: from the reference (run1) or relative to its value at the phase start")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--pool", type=int, default=1, help="average-pool the 16x16 patch grid (2 -> 8x8)")
    ap.add_argument("--drop_agent", type=float, default=0.3)
    ap.add_argument("--d_scale", type=float, default=0.08)
    ap.add_argument("--w_follow", type=float, default=0.3)
    ap.add_argument("--w_base", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    torch.manual_seed(args.seed)
    idx = build_index(args)
    from pace.pi05 import InstructionText

    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(idx["lang"].values()))
    lang_vec = {t: text.embed(l).float() for t, l in idx["lang"].items()}
    dval = np.stack([s["d"] for s in idx["samples"] if s["d"] is not None])
    print(f"[d label] {args.dlabel}: {len(dval)} labelled samples", flush=True)
    print(f"[d] |d| median {100 * np.median(np.linalg.norm(dval, axis=1)):.2f} cm, 99th "
          f"{100 * np.percentile(np.abs(dval), 99):.2f} cm per axis, max {100 * np.abs(dval).max():.2f} cm "
          f"(tanh scale {100 * args.d_scale:.0f} cm)", flush=True)
    tr, va = PairData(idx, "train", args, lang_vec, True), PairData(idx, "val", args, lang_vec, False)
    print(f"[data] train {len(tr)} samples, val {len(va)}", flush=True)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, drop_last=True, **kw)
    vl = torch.utils.data.DataLoader(va, 64, shuffle=False, **kw)
    D = np.load(f"{args.feats}/pca.npz")["comp"].shape[0]
    model = Comparator(D, w=args.width, pool=args.pool).cuda()
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.max_steps or args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step, best, t0 = 0, float("inf"), time.time()
    log = open(f"{args.out}/log.jsonl", "a")
    run = collections.defaultdict(float)
    while step < total:
        for b in tl:
            b = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                d, lf, lb = model(b["feats"], b["state"], b["lang"], b["drop"], args.d_scale)
            dm, fm, bm = b["d_ok"].float(), b["follow_ok"].float(), b["base_ok"].float()
            l_d = (F.smooth_l1_loss(100 * d.float(), 100 * b["d"], beta=0.5, reduction="none").sum(1) * dm).sum() / dm.sum().clamp(min=1)
            l_f = (F.binary_cross_entropy_with_logits(lf.float(), b["follow"], reduction="none") * fm).sum() / fm.sum().clamp(min=1)
            l_b = (F.binary_cross_entropy_with_logits(lb.float(), b["base"], reduction="none") * bm).sum() / bm.sum().clamp(min=1)
            loss = l_d + args.w_follow * l_f + args.w_base * l_b
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in (("loss", loss), ("d_cm", l_d), ("follow", l_f), ("base", l_b)):
                run[k] += float(v)
            if step % 100 == 0:
                print(f"[train] step {step}/{total} ({time.time() - t0:.0f}s) " +
                      " ".join(f"{k} {v / 100:.3f}" for k, v in run.items()), flush=True)
                run.clear()
            if step % args.eval_every == 0 or step == total:
                ev = evaluate(model, vl, args)
                log.write(json.dumps(dict(step=step, **ev)) + "\n")
                log.flush()
                print(f"[val] step {step}: {json.dumps(ev)}", flush=True)
                torch.save(dict(model=model.state_dict(), args=vars(args), step=step, val=ev), f"{args.out}/last.pt")
                if ev["perturbed_all_median"] < best:
                    best = ev["perturbed_all_median"]
                    torch.save(dict(model=model.state_dict(), args=vars(args), step=step, val=ev), f"{args.out}/best.pt")
            if step >= total:
                break
    print(f"-> {args.out} (best perturbed median {best:.3f} cm)", flush=True)


if __name__ == "__main__":
    main()
