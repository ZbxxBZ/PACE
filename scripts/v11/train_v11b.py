"""
V11b (11-02; desktop design doc 10.14f). pi0.5 stays frozen and V11's branch stays (train_v11_v9.Runner, reference isolation off), with
these changes:
  1. Compare: one token per image cell from cur - ref, bias-free, read by every action step directly (v11_imgpath.py's prototype). Its
     output is added to the branch's: Delta = Delta_branch + Delta_compare.
  2. Gate: g in (0, 1) from the compare tokens and cur_state - ref_state, supervised by how the sample was paired (BCE; matched 1,
     mismatched 0). The executed correction is g Delta. No d head.
  3. Reference step: --align pma reads the step found by a state-only PMA (RTCF, arXiv 2608.04527; hand position and gripper, no image,
     nothing learned, v11b_pma_check.py / v11b_pma_events.py); --align time reads the call nearest to t, as V11.
  4. Data, shares per sample: camera (V11's samples, Delta* = clear view - a_pi) 30%; wide (runs/v11b, camera changed as LIBERO-plus
     changes it) 20%; replay (runs/v11b keep + pure, Delta* = 0) 15%; noncam (runs/v11b, successful episodes only, Delta* = 0) 15%;
     other (any of those with another task's reference, Delta* = 0, g = 0) 15%; shift (the true reference read at least --shift steps from
     the aligned step, Delta* = 0, g = 0) 5%. pi0.5's own episodes are 40% of camera, wide and replay.
  5. --drop_act: share of samples whose reference chunk is zeroed in training, so that the branch cannot only follow it.
References have their camera in place, lights kept (runs/v11/refvis_orig). Checkpoint: lowest share-weighted squared error of g Delta.
    python train_v11b.py --out runs/v11b/train/pma --align pma
    python train_v11b.py --out runs/v11b/train/pma --probe last.pt
"""
import argparse
import collections
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_v11_v9 as M

V, V11, CALL, CHUNK, EXEC, FRAME_KEYS = M.V, M.V11, M.CALL, M.CHUNK, M.EXEC, M.FRAME_KEYS
V11B = "/root/pace/runs/v11b"
SHARE = dict(camera=.30, wide=.20, replay=.15, noncam=.15, other=.15, shift=.05)
BASE = ("camera", "wide", "replay", "noncam")
VAL_ONLY = ("negative",)  # V10's negative pairs: only to compare the pull toward the reference with V11's
KIND_SHARE = dict(expert=.2, perturbed=.2, carry=.2, pi05_aug=.4)
KINDS = ("expert", "perturbed", "carry", "pi05_aug", "noncam", "negative")
SP, SG, VMAX, GAMMA = .10, .02, 3, .01
cv2.setNumThreads(0)


def pma(P, G, rp, rg):
    """State-only PMA (pace.core.ProgressiveAlignment's recursion): [calls, 3], [calls, 2] against a reference's -> its call per call."""
    L, J, out = len(rp), None, []
    for i in range(len(P)):
        c = 1.0 - np.exp(-((rp - P[i]) ** 2).sum(-1) / SP ** 2 - ((rg - G[i]) ** 2).sum(-1) / SG ** 2)
        if J is None:
            J = c
        else:
            best = np.full(L, np.inf)
            for d in range(min(VMAX + 1, L)):
                s = J if d == 0 else np.r_[np.full(d, np.inf), J[:-d]]
                best = np.minimum(best, s + GAMMA * abs(d - 1))
            J = c + best
        out.append(int(J.argmin()))
    return np.array(out)


def episodes(args):
    """Every episode as a dict: group, kind, frames, a_pi, clear chunk (None: Delta* = 0), reference, split, task, instruction, call rows."""
    a0 = argparse.Namespace(**json.load(open("runs/v11/train/v9_run2_4090/args.json")))
    kinds = list(a0.kinds)
    a0.kinds, a0.dlabel, a0.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
    idx = M.T.build_index(a0)
    a0.kinds = kinds
    idx = M.add_pi05_aug(idx)
    refs = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
    lang = {r["task_id"]: r["language"] for r in refs}
    have = {f"t{r['task_id']}_i{r['init']}" for r in refs if os.path.exists(f"{args.refvis}/t{r['task_id']}_i{r['init']}_a0.npz")}
    task_of_base = {r["base"]: r["task_id"] for r in refs}
    eps, by_key = [], {}

    def add(**e):
        e["t"] = np.asarray(e["t"])
        e["rows"] = np.where(e["t"] % CALL == 0)[0]
        e["id"] = len(eps)
        eps.append(e)

    for e in idx["eps"]:
        k, fkey = e["kind"], e["fkey"]
        ok = e["ref"] in have and not (k == "expert" and not e["meta"]["success"]) and os.path.exists(f"{V11}/api/{k}/{fkey}.npy")
        by_key[k, fkey] = e
        if ok and k == "negative":
            add(group="negative", kind=k, frames=f"{V}/frames/{k}/{fkey}.npz", api=f"{V11}/api/{k}/{fkey}.npy", clear=None, ref=e["ref"],
                split=e["split"], task=e["task"], lang=lang[e["task"]], t=e["meta"]["t"])
        elif ok and k != "negative" and os.path.exists(f"{V11}/labels_cam/{k}/{fkey}.npy"):
            add(group="camera", kind=k, frames=f"{V}/frames/{k}/{fkey}.npz", api=f"{V11}/api/{k}/{fkey}.npy",
                clear=f"{V11}/labels_cam/{k}/{fkey}.npy", ref=e["ref"], split=e["split"], task=e["task"], lang=lang[e["task"]], t=e["meta"]["t"])
    for m in ("keep", "pure", "wide"):
        for r in map(json.loads, open(f"{V11B}/{m}.jsonl")):
            e = by_key.get((r["kind"], f"{r['key']}_a0"))
            if e is None or e["ref"] not in have:
                continue
            add(group="wide" if m == "wide" else "replay", kind=r["kind"], frames=f"{V11B}/frames/{m}/{r['fkey']}.npz",
                api=f"{V11B}/api/{m}/{r['fkey']}.npy", clear=f"{V11B}/labels/wide/{r['fkey']}.npy" if m == "wide" else None,
                ref=e["ref"], split=e["split"], task=e["task"], lang=lang[e["task"]], t=e["meta"]["t"])
    pool = collections.defaultdict(list)
    for k in sorted(have):
        tid, init = int(k.split("_i")[0][1:]), int(k.split("_i")[1])
        pool[tid, init in M.VAL_INITS].append(k)
    for r in map(json.loads, open(f"{V11B}/noncam.jsonl")):
        tid = task_of_base.get(r["base"])
        if not r["success"] or tid is None or not pool[tid, r["split"] == "val"]:
            continue  # first version: failed episodes stay out of the loss
        p = pool[tid, r["split"] == "val"]
        add(group="noncam", kind="noncam", frames=f"{V11B}/frames/noncam/{r['key']}.npz", api=f"{V11B}/api/noncam/{r['key']}.npy", clear=None,
            ref=p[int(np.random.default_rng([r["task_id"], r["init"], 5]).integers(len(p)))], split=r["split"], task=tid, lang=r["language"],
            t=np.arange(r["calls"]) * CALL)
    print("[data] episodes", sorted(collections.Counter((e["group"], e["split"]) for e in eps).items()), flush=True)
    return eps, refs, have


class Mix(torch.utils.data.Dataset):
    """Training: every index is one sample drawn by SHARE (seeded by the index, so runs see the same sequence). Validation: one group."""

    def __init__(self, eps, refs, have, split, args, n, group=None):
        self.eps, self.args, self.n, self.group, self.split = [e for e in eps], args, n, group, split
        z = np.load(f"{V11}/api/norm.npz")
        self.mean, self.std = z["mean"], z["std"]
        self.ref_file = {f"t{r['task_id']}_i{r['init']}": f"{V}/refs/{r['actions']}" for r in refs}
        ref_task = {f"t{r['task_id']}_i{r['init']}": r["task_id"] for r in refs}
        self.others = collections.defaultdict(list)
        for k in sorted(have):
            for t_ in set(ref_task.values()):
                if ref_task[k] != t_:
                    self.others[t_].append(k)
        self.items = collections.defaultdict(lambda: collections.defaultdict(list))
        for e in eps:
            if e["split"] == split:
                self.items[e["group"]][e["kind"]].extend((e["id"], int(r)) for r in e["rows"])
        self.items = {g: {k: np.array(v) for k, v in d.items()} for g, d in self.items.items()}
        self._cache, self._pma = {}, {}

    def __len__(self):
        return self.n

    def _small(self, path, field=None):
        key = (path, field)
        if key not in self._cache:
            if path.endswith(".npz"):
                with np.load(path) as z:
                    self._cache[key] = {k: np.array(z[k]) for k in z.files} if field is None else np.array(z[field])
            else:
                self._cache[key] = np.load(path)
        return self._cache[key]

    def norm(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    def pick(self, group, rng):
        d = self.items[group]
        ks = sorted(d)
        p = np.array([KIND_SHARE.get(k, 1.0) for k in ks], float)
        a = d[ks[int(rng.choice(len(ks), p=p / p.sum()))]]
        return a[int(rng.integers(len(a)))]

    def step_of(self, e, row, rkey, rt):
        """Index into the reference's call rows for this sample."""
        if self.args.align == "time":
            return int(np.argmin(np.abs(rt - int(e["t"][row]))))
        key = (e["id"], rkey)
        j = self._pma.get(key)
        if j is None:
            if len(self._pma) > 30000:
                self._pma.clear()
            f = f"{V}/frames/refs/{rkey}_a0.npz"
            rows = np.where(self._small(f, "t") % CALL == 0)[0]
            P, G = self._small(e["frames"], "eef_pos")[e["rows"]], self._small(e["frames"], "gripper")[e["rows"]]
            j = self._pma[key] = pma(P, G, self._small(f, "eef_pos")[rows], self._small(f, "gripper")[rows])
        return int(j[np.searchsorted(e["rows"], row)])

    def sample(self, group, rng):
        if group in BASE or group in VAL_ONLY:
            base = group
        else:
            p = np.array([SHARE[g] for g in BASE])
            base = BASE[int(rng.choice(len(BASE), p=p / p.sum()))]
        eid, row = self.pick(base, rng)
        e = self.eps[int(eid)]
        rkey = e["ref"]
        if group == "other":
            p = self.others[e["task"]]
            rkey = p[int(rng.integers(len(p)))]
        rv = self._small(f"{self.args.refvis}/{rkey}_a0.npz")
        calls = np.where(rv["t"] % CALL == 0)[0]
        rt = rv["t"][calls]
        q = self.step_of(e, row, rkey, rt)
        if group == "shift":
            far = np.where(np.abs(rt - rt[q]) >= self.args.shift)[0]
            if not len(far):
                return None
            q = int(far[int(rng.integers(len(far)))])
        j = int(calls[q])
        with np.load(e["frames"]) as fr:
            cur = {k: np.array(fr[k][row]) for k in FRAME_KEYS}
        for k in FRAME_KEYS:
            if k in ("agent", "wrist"):
                if cur[k].shape[0] != 224:  # the non-camera episodes keep the env's frames; pi0.5 resizes them the same way
                    cur[k] = cv2.resize(cur[k], (224, 224), interpolation=cv2.INTER_LINEAR)
            else:
                cur[k] = cur[k].astype(np.float64)
        a_pi = np.asarray(self._small(e["api"])[row], np.float32)
        tgt = np.zeros_like(a_pi)
        if group in ("camera", "wide"):
            tgt = np.asarray(self._small(e["clear"])[row], np.float32) - a_pi
        a_ref = self.norm(M.chunk(self._small(self.ref_file[rkey]).astype(np.float64), int(rv["t"][j])))
        L = len(calls)
        return dict(**cur, lang=e["lang"], t=int(e["t"][row]), a_pi=torch.from_numpy(a_pi), target=torch.from_numpy(tgt),
                    a_ref=torch.from_numpy(a_ref), ref_vis=torch.from_numpy(np.asarray(rv["vis"][j])),
                    ref_state=torch.from_numpy(rv["state"][j].astype(np.float32)),
                    ref_pos=torch.tensor([q / max(L - 1, 1), q, L], dtype=torch.float32), valid=torch.tensor(True),
                    match=torch.tensor(0.0 if group in ("other", "shift", "negative") else 1.0), kind=KINDS.index(e["kind"]),
                    rkey=rkey, rrow=j, eid=int(eid), row=int(row))

    def __getitem__(self, i):
        rng = np.random.default_rng([self.args.seed, i])
        g = self.group or list(SHARE)[int(rng.choice(len(SHARE), p=np.array(list(SHARE.values()))))]
        for _ in range(100):
            s = self.sample(g, rng)
            if s is not None:
                return s
        raise RuntimeError(f"no sample for {g}")


class DeltaHead(nn.Module):
    """train_v11_v9.DiffHead without d; keeps h_off for the compare head."""

    def __init__(self, dim=1024, hidden=512, action_dim=7, init_std=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, action_dim, bias=False)
        nn.init.normal_(self.fc2.weight, std=init_std)
        self.h_off = None

    def forward(self, h_on, h_off):
        self.h_off = h_off
        diff = (h_on - h_off) / (h_off.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return self.fc2(F.gelu(self.fc1(diff))), torch.zeros(h_on.shape[0], 2, device=h_on.device)


class Compare(nn.Module):
    """One token per image cell from cur - ref, bias-free (exactly 0 where the two agree), read by every action step; output starts at 0."""

    def __init__(self, vis_dim=2048, d=256, heads=4, tokens=32, dim=1024, action_dim=7):
        super().__init__()
        self.heads = heads
        self.fc1, self.fc2 = nn.Linear(vis_dim, d, bias=False), nn.Linear(d, d, bias=False)
        self.pos = nn.Parameter(torch.randn(tokens, d) * 0.02)
        self.q = nn.Linear(dim, d)
        self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o1, self.o2 = nn.Linear(d, d, bias=False), nn.Linear(d, action_dim, bias=False)
        nn.init.zeros_(self.o2.weight)

    def tokens(self, cur, ref):
        return self.fc2(F.gelu(self.fc1((cur - ref) * 32))) * (1 + self.pos)  # cells are unit vectors of 2048 numbers

    def forward(self, c, h):
        B, N, H = h.shape[0], h.shape[1], self.heads
        q = self.q(h).view(B, N, H, -1).transpose(1, 2)
        k = self.k(c).view(B, c.shape[1], H, -1).transpose(1, 2)
        v = self.v(c).view(B, c.shape[1], H, -1).transpose(1, 2)
        return self.o2(F.gelu(self.o1(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1))))


class Gate(nn.Module):
    """Is this reference the right one here? From the compare tokens and the state difference."""

    def __init__(self, d=256, state_dim=8, hidden=256):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(2 * d + state_dim, hidden), nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 1.5)  # g starts near 0.8

    def forward(self, c, ds):
        return torch.sigmoid(self.fc2(F.gelu(self.fc1(torch.cat([c.mean(1), c.abs().amax(1), ds], -1)))))[:, 0]


class Runner(M.Runner):
    def __init__(self, args):
        super().__init__(args)
        self.head, self.cmp, self.gate = DeltaHead().to(self.device), Compare().to(self.device), Gate().to(self.device)
        self.seen = {}
        vt, st = self.vis_tokens, self.batch_state

        def vis(*a, **k):
            self.seen["cur"] = vt(*a, **k)
            return self.seen["cur"]

        def state(*a, **k):
            self.seen["state"] = st(*a, **k)
            return self.seen["state"]

        self.vis_tokens, self.batch_state = vis, state

    def step(self, b, drop=0.0, g_from=None):
        """-> Delta, its two parts and g. g_from: another batch's reference images and states for the gate only (probe)."""
        dev = self.device
        if drop:
            m = torch.rand(len(b["lang"])) < drop
            b = dict(b, a_ref=b["a_ref"].masked_fill(m[:, None, None], 0.0))
        d_b, _, stats = M.Runner.__call__(self, b)
        cur, st = self.seen["cur"].float(), self.seen["state"].float()
        c = self.cmp.tokens(cur, b["ref_vis"].to(dev).float())
        d_c = self.cmp(c, self.head.h_off)
        src = g_from if g_from is not None else b
        c_g = c if g_from is None else self.cmp.tokens(cur, src["ref_vis"].to(dev).float())
        g = self.gate(c_g, st - src["ref_state"].to(dev).float())
        return dict(delta=d_b + d_c, branch=d_b, compare=d_c, g=g, stats=stats)

    def groups(self):
        new = list(self.head.parameters()) + list(self.cmp.parameters()) + list(self.gate.parameters())
        return [p for p in self.adapter.parameters() if p.requires_grad], new

    def modes(self, train):
        for m in (self.adapter, self.head, self.cmp, self.gate):
            m.train(train)

    def save(self, path, **extra):
        torch.save(dict(adapter=self.adapter.state_dict(), config=self.adapter.config, head=self.head.state_dict(), cmp=self.cmp.state_dict(),
                        gate=self.gate.state_dict(), args=vars(self.args), **extra), path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.adapter.load_state_dict(ck["adapter"])
        self.head.load_state_dict(ck["head"]); self.cmp.load_state_dict(ck["cmp"]); self.gate.load_state_dict(ck["gate"])
        return ck


def rms(x):
    return (x[:, :EXEC] ** 2).mean((1, 2)).sqrt()


def auc(pos, neg):
    x = np.r_[pos, neg]
    r = np.empty(len(x))
    o = np.argsort(x)
    r[o] = np.arange(1, len(x) + 1)
    for v in np.unique(x[np.diff(np.sort(x), prepend=np.nan) == 0]):
        r[x == v] = r[x == v].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pull(d, c):
    """Share of the energy of d along the copy direction c (a_ref - a_pi), pooled, as v11_copy.py."""
    return float((d * c).sum() ** 2 / ((c * c).sum() * (d * d).sum() + 1e-12))


@torch.no_grad()
def evaluate(run, loaders):
    run.modes(False)
    out, gs, score = {}, {}, 0.0
    for g, loader in loaders.items():
        R = collections.defaultdict(list)
        for b in loader:
            r = run.step(b)
            tgt, x = b["target"].to(run.device), r["g"][:, None, None] * r["delta"]
            R["err"].append(rms(x - tgt).cpu()); R["zero"].append(rms(tgt).cpu()); R["out"].append(rms(x).cpu())
            R["raw"].append(rms(r["delta"]).cpu()); R["cmp"].append(rms(r["compare"]).cpu()); R["branch"].append(rms(r["branch"]).cpu())
            R["g"].append(r["g"].cpu()); R["kind"].append(b["kind"]); R["sq"].append(((x - tgt)[:, :EXEC] ** 2).mean((1, 2)).cpu())
        r = {k: torch.cat(v).float().numpy() for k, v in R.items()}
        gs[g] = r["g"]
        for k in ("err", "zero", "out", "raw", "cmp", "branch", "g"):
            out[f"{g}_{k}"] = round(float(r[k].mean()), 4)
        if g in ("camera", "wide"):
            for i, nm in enumerate(KINDS):
                s = r["kind"] == i
                if s.any():
                    out[f"{g}_{nm}_err"], out[f"{g}_{nm}_zero"] = round(float(r["err"][s].mean()), 4), round(float(r["zero"][s].mean()), 4)
        if g in SHARE:
            score += SHARE[g] * float(r["sq"].mean())
    good = np.concatenate([gs[g] for g in BASE])
    out["auc_other"], out["auc_shift"] = round(auc(good, gs["other"]), 4), round(auc(good, gs["shift"]), 4)
    out["score"] = round(score, 5)
    run.modes(True)
    return out


@torch.no_grad()
def probe(run, va, args, name):
    """After training, on validation: which inputs matter. Swapped inputs come from the next sample of a shuffled batch (v11_paths.py)."""
    run.modes(False)
    res = dict(ckpt=name, step=run.load(f"{args.out}/{name}").get("step"))
    D = collections.defaultdict(list)
    for g in ("camera", "wide"):
        for b in torch.utils.data.DataLoader(va[g], 16, shuffle=True, num_workers=2, drop_last=True, generator=torch.Generator().manual_seed(0)):
            tgt = b["target"].to(run.device)
            r = run.step(b)
            full = r["g"][:, None, None] * r["delta"]
            D["full"].append(rms(full - tgt).cpu()); D["zero"].append(rms(tgt).cpu()); D["kind"].append(b["kind"])
            sw = dict(b, ref_vis=torch.roll(b["ref_vis"], 1, 0))
            r2 = run.step(sw)  # images swapped in Delta and in the gate
            D["swap_vis_gate"].append(rms(r2["g"][:, None, None] * r2["delta"] - tgt).cpu())
            D["swap_vis"].append(rms(r["g"][:, None, None] * r2["delta"] - tgt).cpu())  # gate held: only the path into Delta
            D["chg_vis"].append((rms(r2["delta"] - r["delta"]) / (rms(r["delta"]) + 1e-9)).cpu())
            sw = dict(b, a_ref=torch.roll(b["a_ref"], 1, 0))
            D["swap_act"].append(rms(r["g"][:, None, None] * run.step(sw)["delta"] - tgt).cpu())
            D["no_compare"].append(rms(r["g"][:, None, None] * r["branch"] - tgt).cpu())
            D["no_branch"].append(rms(r["g"][:, None, None] * r["compare"] - tgt).cpu())
    d = {k: torch.cat(v).float().numpy() for k, v in D.items()}
    rng = np.random.default_rng(0)
    pick = rng.integers(0, len(d["full"]), (10000, len(d["full"])))
    res["n"], res["zero"], res["full"] = len(d["full"]), float(d["zero"].mean()), float(d["full"].mean())
    own = d["kind"] == KINDS.index("pi05_aug")
    res["pi05_own"] = dict(n=int(own.sum()), zero=float(d["zero"][own].mean()), full=float(d["full"][own].mean()))
    res["change_of_delta_with_swapped_images"] = float(d["chg_vis"].mean())
    for k in ("swap_vis", "swap_vis_gate", "swap_act", "no_compare", "no_branch"):
        e = d[k] - d["full"]
        lo, hi = np.percentile(e[pick].mean(1), [2.5, 97.5])
        res[k] = dict(err=float(d[k].mean()), rise=float(e.mean()), lo=float(lo), hi=float(hi), rise_pi05_own=float(e[own].mean()))
    for g in ("other", "negative"):
        X, C = [], []
        for b in torch.utils.data.DataLoader(va[g], 16, num_workers=2):
            r = run.step(b)
            X.append((r["g"][:, None, None] * r["delta"])[:, :EXEC].cpu().float())
            C.append((b["a_ref"] - b["a_pi"])[:, :EXEC].float())
        X, C = torch.cat(X).numpy(), torch.cat(C).numpy()
        res[f"pull_{g}"] = dict(share=pull(X, C), rms=float(np.sqrt((X ** 2).mean())))
    res["val"] = evaluate(run, {g: torch.utils.data.DataLoader(va[g], 16, num_workers=2) for g in va})
    json.dump(res, open(f"{args.out}/probe.json", "w"), indent=1)
    print("[probe]", json.dumps(res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--align", choices=("pma", "time"), default="pma")
    ap.add_argument("--refvis", default=f"{V11}/refvis_orig")
    ap.add_argument("--max_refs", type=int, default=4)
    ap.add_argument("--read_time", type=float, default=0.05)
    ap.add_argument("--ref_isolation", choices=("on", "off"), default="off")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4, help="the branch")
    ap.add_argument("--lr_new", type=float, default=3e-4, help="compare, gate and head, which start from nothing")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--w_exec", type=float, default=1000.0)
    ap.add_argument("--w_gate", type=float, default=0.1)
    ap.add_argument("--drop_act", type=float, default=0.4)
    ap.add_argument("--shift", type=int, default=150, help="steps between the read step and the aligned step of a shift sample")
    ap.add_argument("--val_n", type=int, default=300, help="validation samples per group")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=0, help="keep a checkpoint at these steps (multiples of --eval_every)")
    ap.add_argument("--ramp_from", type=int, default=0, help="continuing a finished run: the step it ended at, the learning rate rises from 0 there")
    ap.add_argument("--ramp", type=int, default=500, help="steps of that rise")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="20 steps, 16 validation samples per group")
    ap.add_argument("--probe", default=None, help="only the probe of this checkpoint in --out")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.val_n, args.eval_every, args.warmup = 20, 16, 20, 5
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    eps, refs, have = episodes(args)
    va = {g: Mix(eps, refs, have, "val", args, args.val_n, group=g) for g in (*SHARE, *VAL_ONLY)}
    vls = {g: torch.utils.data.DataLoader(va[g], args.bs, num_workers=2, persistent_workers=True) for g in SHARE}
    run = Runner(args)
    if args.probe:
        return probe(run, va, args, args.probe)
    tr = Mix(eps, refs, have, "train", args, args.steps * args.accum * args.bs)
    print("[data] train samples", {g: {k: len(v) for k, v in d.items()} for g, d in tr.items.items()}, flush=True)
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=False, drop_last=True, num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    old, new = run.groups()
    print(f"[model] branch {sum(p.numel() for p in old) / 1e6:.1f}M, compare + gate + head {sum(p.numel() for p in new) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW([dict(params=old, lr=args.lr), dict(params=new, lr=args.lr_new)], weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))
        * (min(1.0, (s - args.ramp_from + 1) / args.ramp) if args.ramp_from and s >= args.ramp_from else 1.0))
    W = M.weights(args, run.device)
    log = open(f"{args.out}/log.jsonl", "a")
    step, best = 0, float("inf")
    if os.path.exists(f"{args.out}/last.pt") and not args.smoke:  # a stopped run goes on from its last validation
        ck = run.load(f"{args.out}/last.pt")
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        step, best = ck["step"], ck["best"]
        print(f"[resume] step {step}", flush=True)
    else:
        ev = evaluate(run, vls)
        print(f"[val] step 0: {json.dumps(ev)}", flush=True)
        log.write(json.dumps(dict(step=0, **ev)) + "\n"); log.flush()
    t0, it = time.time(), iter(tl)
    acc = collections.defaultdict(float)
    run.modes(True)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            b = next(it)
            r = run.step(b, drop=args.drop_act)
            tgt, m = b["target"].to(run.device), b["match"].to(run.device)
            l_res = (((r["g"][:, None, None] * r["delta"] - tgt) ** 2).mean(-1) * W).sum(-1).mean()
            l_g = F.binary_cross_entropy(r["g"], m)
            ((l_res + args.w_gate * l_g) / args.accum).backward()
            acc["res"] += float(l_res.detach()) / args.accum
            acc["bce"] += float(l_g.detach()) / args.accum
            acc["cmp"] += float(rms(r["compare"].detach()).mean()) / args.accum
            acc["branch"] += float(rms(r["branch"].detach()).mean()) / args.accum
        torch.nn.utils.clip_grad_norm_(old, 1.0)
        torch.nn.utils.clip_grad_norm_(new, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0 or args.smoke:
            n = 1 if args.smoke else 50
            print(f"[train] step {step}/{args.steps} ({time.time() - t0:.0f}s) " + " ".join(f"{k} {v / n:.4f}" for k, v in acc.items()), flush=True)
            acc.clear()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(run, vls)
            log.write(json.dumps(dict(step=step, **ev)) + "\n"); log.flush()
            print(f"[val] step {step}: {json.dumps(ev)}", flush=True)
            if ev["score"] < best:
                best = ev["score"]
                run.save(f"{args.out}/best.pt", step=step, val=ev)
            run.save(f"{args.out}/last.pt", step=step, val=ev, best=best, opt=opt.state_dict(), sched=sched.state_dict())
            if args.save_every and step % args.save_every == 0:
                run.save(f"{args.out}/step_{step}.pt", step=step, val=ev)
    print(f"-> {args.out} (best validation score {best:.5f})", flush=True)
    probe(run, va, args, "last.pt")


if __name__ == "__main__":
    main()
