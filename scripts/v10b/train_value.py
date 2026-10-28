"""
V10-3 value head (10-25; docs/PACE_V1-V8.md "下一步路线"). On top of the FROZEN V10-3 backbone (train_v10b.py run3:
+ carry pairs, hand offset relative to the phase start), a small MLP with two outputs at a decision point:
  P(cont) : pi0.5 keeps control from here and the episode succeeds;
  P(take) : the reference phase k is followed from its start (and later decisions as the data's policy made them),
            and the episode succeeds.
Decision points are at the start of a reference phase (step 0, the end of a followed phase, after one of pi0.5's
grasps / releases, an empty grasp / being stuck), so the hand offset relative to the phase start is 0 there.
Data, training scenes only (init states 6..44 train / 45..49 validation; never a test variant):
  explore : closed_loop_explore.py -- random decisions (start / boundary 0.5, pi0.5 event 0.3, trigger 0.5), the
            network inputs saved at each decision; label = the episode's outcome, on the output of the choice made;
  pi05    : pi0.5's own augmented rollouts (collect_pi05.py --set orig_aug, frames + features rendered): every 3rd call,
            phase k = pi0.5's gripper command changes so far, a reference of the same task / split; label = outcome, on
            P(cont) only.
Inputs: the backbone's pooled queries h (8 x width), its d (norm) and old heads' probabilities, the decision kind
(start / boundary / event / trigger), phase k / number of phases, tau / 600. Loss: BCE on the observed output, each
output's positives / negatives re-weighted to balance. Temperature per output fitted on validation.
Validation report: AUROC per output (and per data source), and an inverse-propensity estimate of the episode success of
"take iff P(take) - P(cont) > m" on the explore validation decisions, vs always / never taking and the random policy.
    python train_value.py --backbone runs/v10b/train/run3/best.pt --out runs/v10b/value/run1
"""
import argparse
import collections
import glob
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v10b as T
import v10_oracle_shift2 as O

V = "/root/pace/runs/v10b"
VAL = set(range(45, 50))
KINDS = ("start", "boundary", "event", "trigger")
CALL = 10


def kind_id(k):
    return {"start": 0, "boundary": 1, "event": 2}.get(k, 3)  # empty / stuck -> trigger


def load_refs():
    refs_rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
    recs = pickle.load(open(f"{V}/pairs/refs_rec.pkl", "rb"))
    out = {}
    for r in refs_rows:
        key = f"t{r['task_id']}_i{r['init']}"
        rec = recs.get((r["task_id"], r["init"]))
        if not (r["success"] and rec and rec["success"] and os.path.exists(f"{V}/feats/refs/{key}_a0.npy")):
            continue
        fm = np.load(f"{V}/frames/refs/{key}_a0.npz")
        out[key] = dict(task=r["task_id"], init=r["init"], lang=r["language"], t=fm["t"], eef=fm["eef_pos"],
                        grip=fm["gripper"], phases=[(int(a), int(b)) for a, b, _ in O.phase_targets(rec)],
                        feats=np.load(f"{V}/feats/refs/{key}_a0.npy", mmap_mode="r"))
    return out


def ref_inputs(ref, tau0, ph):
    """Reference rows and state for a decision at the start (tau0) of reference phase ph, offset relative to the phase
    start = 0 (run3's --state phase)."""
    s, e = ref["phases"][ph]
    rows = [int(np.argmin(np.abs(ref["t"] - w))) for w in (tau0 - CALL, tau0, tau0 + CALL, tau0 + 2 * CALL, e)]
    return rows, s, e


def collect(args, refs):
    """Decision samples: dict(cur [2,256,D] fp16, ref key, rows, state [9], lang tasks, kind, ph_frac, tau, out, y, split, src)."""
    S = []
    # explore decisions
    ex = [json.loads(l) for l in open(f"{args.explore}/episodes.jsonl")] if os.path.exists(f"{args.explore}/episodes.jsonl") else []
    for r in ex:
        f = f"{args.explore}/decisions/{r['key']}.npz"
        rkey = f"t{r['ref_task']}_i{r['ref_init']}"
        if not os.path.exists(f) or rkey not in refs:
            continue
        with np.load(f) as zf:  # read once and close (1320 + 440 files: keep no handle open)
            z = {k_: zf[k_] for k_ in zf.files}
        ref = refs[rkey]
        for i in range(len(z["go"])):
            st = z["state"][i].copy()
            if args.state == "phase":
                st[:3] = 0.0
            ph = int(z["ph"][i])
            S.append(dict(cur=z["cur"][i], ref=rkey, rows=z["rows"][i], state=st, tasks=(r["task_id"], ref["task"]),
                          kind=kind_id(str(z["kind"][i])), ph_frac=ph / max(len(ref["phases"]), 1), tau=int(z["tau"][i]),
                          out=1 if bool(z["go"][i]) else 0, y=float(r["success"]), split=r["split"], src="explore",
                          p=(0.3 if str(z["kind"][i]) == "event" else 0.5), ep=r["key"]))
    n_ex = len(S)
    # pi0.5's own rollouts -> P(cont)
    rows_pi = [json.loads(l) for l in open(f"{V}/pi05_aug/episodes.jsonl")]
    byt = collections.defaultdict(list)
    for k_, ref in refs.items():
        byt[ref["task"]].append(k_)
    for r in rows_pi:
        key = f"t{r['task_id']}_i{r['init']}_a{r['aug_idx']}"
        ff, fr = f"{V}/feats/pi05_aug/{key}_a0.npy", f"{V}/frames/pi05_aug/{key}_a0.npz"
        if not (os.path.exists(ff) and os.path.exists(fr)):
            continue
        split = "val" if r["init"] in VAL else "train"
        rng = np.random.default_rng([r["task_id"], r["init"], r["aug_idx"], 5])
        pool = [k_ for k_ in byt[r["task_id"]] if refs[k_]["init"] != r["init"] and ((refs[k_]["init"] in VAL) == (split == "val"))]
        if not pool:
            continue
        rkey = pool[int(rng.integers(len(pool)))]
        ref = refs[rkey]
        acts = np.load(f"{V}/pi05_aug/{r['actions']}")
        closed = acts[:, 6] > 0
        changes, cur_c, pend = [], bool(closed[0]), 0
        for t in range(len(closed)):  # pi0.5's confirmed gripper command changes (3 steps)
            pend = pend + 1 if closed[t] != cur_c else 0
            if pend >= 3:
                cur_c, pend = closed[t], 0
                changes.append(t)
        with np.load(fr) as zf:
            fm = {k_: zf[k_] for k_ in zf.files}
        feats = np.load(ff, mmap_mode="r")
        for c in range(0, len(fm["t"]), 3):
            t = int(fm["t"][c])
            ph = min(sum(1 for x in changes if x <= t), len(ref["phases"]) - 1)
            tau0 = ref["phases"][ph][0]
            rws, s, e = ref_inputs(ref, tau0, ph)
            j = rws[1]
            off = np.zeros(3) if args.state == "phase" else fm["eef_pos"][c] - ref["eef"][j]
            st = np.r_[off, fm["gripper"][c], ref["grip"][j], 0.0, tau0 / 600.0].astype(np.float32)
            S.append(dict(cur=np.array(feats[c]), ref=rkey, rows=np.array(rws), state=st, tasks=(r["task_id"], ref["task"]),
                          kind=0 if c == 0 else 2, ph_frac=ph / max(len(ref["phases"]), 1), tau=tau0, out=0,
                          y=float(r["success"]), split=split, src="pi05", p=1.0, ep=key))
        del feats  # closes the memmap
    print(f"[data] {len(S)} decision samples: explore {n_ex}, pi05 {len(S) - n_ex}", flush=True)
    return S


@torch.no_grad()
def embed(model, S, refs, lang_vec, d_scale, bs=64):
    """Frozen backbone: pooled queries h + its d and old heads at every decision sample."""
    H, X = [], []
    for i in range(0, len(S), bs):
        b = S[i:i + bs]
        cur = torch.from_numpy(np.stack([x["cur"] for x in b]).astype(np.float32))[:, None]
        ref = torch.from_numpy(np.stack([np.asarray(refs[x["ref"]]["feats"][x["rows"]]) for x in b]).astype(np.float32))
        feats = torch.cat([cur, ref], 1).cuda()
        st = torch.from_numpy(np.stack([x["state"] for x in b])).cuda()
        lang = torch.stack([torch.stack([lang_vec[x["tasks"][0]], lang_vec[x["tasks"][1]]]) for x in b]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            d, lf, lb, h = model(feats, st, lang, torch.zeros(len(b), dtype=torch.bool).cuda(), d_scale, return_h=True)
        H.append(h.float().cpu())
        X.append(torch.stack([torch.sigmoid(lf.float()), torch.sigmoid(lb.float()), 100 * d.float().norm(dim=1)], 1).cpu())
    return torch.cat(H), torch.cat(X)


class ValueHead(nn.Module):
    def __init__(self, dim_h, width=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim_h + 3 + len(KINDS) + 2, width), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(width, width), nn.GELU(), nn.Dropout(drop), nn.Linear(width, 2))

    def forward(self, h, x, kind, ph_frac, tau):
        k = F.one_hot(kind, len(KINDS)).float()
        return self.net(torch.cat([h, x, k, ph_frac[:, None], tau[:, None] / 600.0], 1))  # [B, 2] logits: cont, take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=f"{V}/train/run3/best.pt")
    ap.add_argument("--explore", default=f"{V}/closed/explore")
    ap.add_argument("--out", default=f"{V}/value/run1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ck = torch.load(args.backbone, weights_only=False)
    a = ck["args"]
    args.state = a.get("state", "abs")
    D = np.load(f"{V}/feats/pca.npz")["comp"].shape[0]
    model = T.Comparator(D, w=a["width"], pool=a["pool"]).cuda().eval()
    model.load_state_dict(ck["model"])
    refs = load_refs()
    from pace.pi05 import InstructionText

    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned",
                                           tasks=list(dict.fromkeys(r["lang"] for r in refs.values())))
    lang_vec = {r["task"]: text.embed(r["lang"]).float() for r in refs.values()}
    t0 = time.time()
    S = collect(args, refs)
    H, X = embed(model, S, refs, lang_vec, a["d_scale"])
    print(f"[embed] {H.shape} ({time.time() - t0:.0f}s)", flush=True)
    kind = torch.tensor([x["kind"] for x in S])
    phf = torch.tensor([x["ph_frac"] for x in S], dtype=torch.float32)
    tau = torch.tensor([x["tau"] for x in S], dtype=torch.float32)
    out = torch.tensor([x["out"] for x in S])
    y = torch.tensor([x["y"] for x in S], dtype=torch.float32)
    tr = torch.tensor([x["split"] == "train" for x in S])
    va = ~tr
    w = torch.ones(len(S))
    for o in (0, 1):  # balance positives / negatives within each output
        for v in (0.0, 1.0):
            m = tr & (out == o) & (y == v)
            if m.sum():
                w[m] = (tr & (out == o)).sum().float() / (2 * m.sum().float())
    head = ValueHead(H.shape[1]).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    dev = lambda z: z.cuda()
    Hc, Xc, kc, pc, tc, oc, yc, wc = map(dev, (H, X, kind, phf, tau, out, y, w))
    idx_tr = torch.where(tr)[0].cuda()
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        head.train()
        perm = idx_tr[torch.randperm(len(idx_tr), device="cuda")]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            lg = head(Hc[b], Xc[b], kc[b], pc[b], tc[b]).gather(1, oc[b][:, None])[:, 0]
            loss = (F.binary_cross_entropy_with_logits(lg, yc[b], reduction="none") * wc[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            iv = torch.where(va.cuda())[0]
            lg = head(Hc[iv], Xc[iv], kc[iv], pc[iv], tc[iv]).gather(1, oc[iv][:, None])[:, 0]
            vl = float(F.binary_cross_entropy_with_logits(lg, yc[iv]))
        if vl < best:
            best, best_state = vl, {k_: v.detach().clone() for k_, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"[train] epoch {ep} val bce {vl:.4f}", flush=True)
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        L = head(Hc, Xc, kc, pc, tc).cpu()
    # temperature per output on validation
    temps = []
    for o in (0, 1):
        m = va & (out == o)
        best_t, best_l = 1.0, 1e9
        for tt in np.linspace(0.5, 4.0, 36):
            l_ = float(F.binary_cross_entropy_with_logits(L[m, o] / tt, y[m])) if m.sum() else 0.0
            if l_ < best_l:
                best_t, best_l = float(tt), l_
        temps.append(best_t)
    P = torch.sigmoid(L / torch.tensor(temps))
    rep = dict(n=len(S), temps=temps)
    for o, nm in ((0, "cont"), (1, "take")):
        for src in ("explore", "pi05", "all"):
            m = va & (out == o) & (torch.tensor([src == "all" or x["src"] == src for x in S]))
            if m.sum() > 10:
                rep[f"auroc_{nm}_{src}"] = round(T.auroc(y[m].numpy() > 0.5, P[m, o].numpy()), 3)
                rep[f"n_{nm}_{src}"] = int(m.sum())
    # IPS estimate on explore validation decisions: value of "take iff P(take) - P(cont) > m"
    ev = [i for i, x in enumerate(S) if x["split"] == "val" and x["src"] == "explore"]
    if ev:
        ips = {}
        for mth in (-0.2, -0.1, 0.0, 0.1, 0.2):
            num, cnt = 0.0, 0
            for i in ev:
                x = S[i]
                want = int(float(P[i, 1] - P[i, 0]) > mth)
                pr = x["p"] if x["out"] == 1 else 1 - x["p"]
                if want == x["out"]:
                    num += x["y"] / pr
                cnt += 1
            ips[f"m{mth:+.1f}"] = round(num / cnt, 3)
        for pol, want_fn in (("always_take", lambda x: 1), ("never_take", lambda x: 0)):
            num = sum(S[i]["y"] / (S[i]["p"] if S[i]["out"] == 1 else 1 - S[i]["p"]) for i in ev if want_fn(S[i]) == S[i]["out"])
            ips[pol] = round(num / len(ev), 3)
        ips["random(logged)"] = round(float(np.mean([S[i]["y"] for i in ev])), 3)
        rep["ips_val_decisions"] = ips
        rep["n_val_decisions"] = len(ev)
    print("[report]", json.dumps(rep), flush=True)
    torch.save(dict(head=head.state_dict(), temps=temps, dim_h=H.shape[1], backbone=args.backbone, report=rep),
               f"{args.out}/value.pt")
    json.dump(rep, open(f"{args.out}/report.json", "w"), indent=1)
    print(f"-> {args.out}/value.pt", flush=True)


if __name__ == "__main__":
    main()
