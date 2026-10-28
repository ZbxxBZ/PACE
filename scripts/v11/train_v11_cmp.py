"""
V11 (10-27; desktop design doc 10.14c, docs/PACE_V1-V8.md "V11"): the experience branch predicts a residual on
pi0.5's action chunk, a = a_pi + Delta. pi0.5 stays frozen and is NOT run during training: its chunk a_pi at every
training call is precomputed (v11_check.py --dump -> runs/v11/api/<kind>/<fkey>.npy, normalized action space, one fixed
noise seed per call), so training costs about what V10's comparator did.
One sample = one pi0.5 call t (t % 10 == 0) of a V10 pair episode:
  experience : V10's comparator (train_v10b.Comparator, initialized from runs/v10b/train/run1) reads the current
               frames against the reference's frames at the aligned steps (j = t) and the phase end, the hand offset
               from the reference and both instructions -> pooled queries h and its d estimate;
  chunk      : per step k of the chunk, pi0.5's a_pi[k] and the reference's aligned action a_ref[t + k] (normalized);
  output     : a small transformer over [context(h, d), 50 step tokens] -> Delta[k] (zero-initialized: off = pi0.5),
               times a gate sigmoid(g(h)).
Labels: expert pairs Delta* = normalize(a*[t .. t+49]) - a_pi, a* = the executed reference + true-d controller (after the
end: hold, last gripper command); negative pairs (a reference that should not be followed) Delta* = 0. Loss: squared
error per step, the executed part (first 10 steps) weighted --w_exec; + --w_d x V10's d loss on the backbone (true d).
Validation (init states 45-49, their own references), per sample RMS of the executed part:
  trivial answers -- Delta = 0 (pi0.5 alone), Delta = a_ref - a_pi (copy the reference);
  experience controls -- the same samples with the reference of ANOTHER task, and with NO reference (reference
  features, reference actions and reference instruction zeroed). Registered stop rule: if these are not clearly worse
  than the true reference, the branch is not using the experience.
    python train_v11_cmp.py --out runs/v11/train/cmp_run1   (the comparator variant: a cheap control for the V9-structure V11, train_v11_v9.py)
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

import train_v10b as T

V = "/root/pace/runs/v10b"
V11 = "/root/pace/runs/v11"
CALL, CHUNK, EXEC = 10, 50, 10


def chunk(actions, t):
    """Actions t .. t + CHUNK - 1 of an episode; past the end: hold still with the last gripper command."""
    n = len(actions)
    return np.stack([actions[t + k] if t + k < n else np.r_[np.zeros(6), actions[-1][6]] for k in range(CHUNK)])


class V11Data(torch.utils.data.Dataset):
    """V10's pair samples at pi0.5 calls, with a_pi, a_ref and the residual label. ref_mode: true / other / none."""

    def __init__(self, idx, split, args, lang_vec, train, ref_mode="true"):
        self.idx, self.args, self.train, self.lang_vec, self.ref_mode = idx, args, train, lang_vec, ref_mode
        z = np.load(f"{args.api}/norm.npz")
        self.mean, self.std = z["mean"], z["std"]
        self.items = []
        for i, s in enumerate(idx["samples"]):
            ep = idx["eps"][s["ep"]]
            if ep["kind"] == "expert" and not ep["meta"]["success"]:
                continue  # a failed expert replay: its executed actions are not a correct answer a*
            if (ep["split"] == split and s["t"] % CALL == 0 and os.path.exists(f"{args.api}/{ep['kind']}/{ep['fkey']}.npy")
                    and (ep["kind"] != "expert" or args.label != "clear"
                         or os.path.exists(f"{args.labels}/expert/{ep['fkey']}.npy"))):
                self.items.append(i)
        refs_rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
        self.ref_file = {f"t{r['task_id']}_i{r['init']}": f"{V}/refs/{r['actions']}" for r in refs_rows}
        self.ref_task = {f"t{r['task_id']}_i{r['init']}": r["task_id"] for r in refs_rows}
        self.by_task = collections.defaultdict(list)
        for k in sorted(idx["refs"]):
            self.by_task[self.ref_task[k]].append(k)
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _np(self, path, field=None):
        """Small arrays cached per worker as plain arrays (no open files kept: mmaps / npz handles hold a descriptor)."""
        if path not in self._cache:
            if field is None:
                self._cache[path] = np.load(path)
            else:
                with np.load(path) as z:
                    self._cache[path] = np.array(z[field])
        return self._cache[path]

    def norm(self, a):
        return ((a - self.mean) / self.std).astype(np.float32)

    def __getitem__(self, i):
        s = self.idx["samples"][self.items[i]]
        ep = self.idx["eps"][s["ep"]]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.items[i])
        rkey = ep["ref"]
        if self.ref_mode == "other":  # a reference of another task (deterministic per sample on validation)
            others = [k for t_, ks in sorted(self.by_task.items()) if t_ != ep["task"] for k in ks]
            rkey = others[int(rng.integers(len(others)))]
        refs = self.idx["refs"][rkey]
        a = int(rng.choice(sorted(refs))) if self.train else min(refs)
        rm, t = refs[a], s["t"]
        rows = [int(np.argmin(np.abs(rm["t"] - w))) for w in (t - CALL, t, t + CALL, t + 2 * CALL, s["end"])]
        a_np = np.load(f"{self.args.feats}/{ep['kind']}/{ep['fkey']}.npy", mmap_mode="r")
        cur = torch.from_numpy(np.ascontiguousarray(a_np[[s["c"]]])).float()
        r_np = np.load(f"{self.args.feats}/refs/{rkey}_a{a}.npy", mmap_mode="r")
        ref = torch.from_numpy(np.ascontiguousarray(r_np[rows])).float()
        m, j = ep["meta"], rows[1]
        off = m["eef"][s["c"]] - rm["eef"][j]
        prog = (t - s["start"]) / max(s["end"] - s["start"], 1)
        state = np.r_[off, m["grip"][s["c"]], rm["grip"][j], min(prog, 1.5), t / 600.0]
        a_ref = self.norm(chunk(self._np(self.ref_file[rkey]).astype(np.float64), t))
        lang = torch.stack([self.lang_vec[ep["task"]], self.lang_vec[self.ref_task[rkey]]])
        if self.ref_mode == "none":  # state: [hand offset 3, gripper 2, reference gripper 2, progress, t]
            ref, a_ref = torch.zeros_like(ref), np.zeros_like(a_ref)
            state[:3], state[5:7] = 0.0, 0.0
            lang[1] = 0.0
        a_pi = np.asarray(self._np(f"{self.args.api}/{ep['kind']}/{ep['fkey']}.npy")[s["c"]], np.float32)
        if ep["kind"] == "expert" and self.args.label == "clear":  # what pi0.5 would do seeing clearly (same seed)
            target = np.asarray(self._np(f"{self.args.labels}/expert/{ep['fkey']}.npy")[s["c"]], np.float32) - a_pi
        elif ep["kind"] == "expert":
            acts = self._np(f"{V}/pairs/expert/{ep['key']}.npz", "actions").astype(np.float64)
            target = self.norm(chunk(acts, t)) - a_pi
        else:  # negative: this reference should not be followed -> leave pi0.5 alone
            target = np.zeros_like(a_pi)
        drop = self.train and rng.random() < self.args.drop_agent
        return dict(feats=torch.cat([cur, ref]), state=torch.tensor(state, dtype=torch.float32), drop=torch.tensor(drop),
                    lang=lang, a_pi=torch.from_numpy(a_pi), a_ref=torch.from_numpy(a_ref),
                    target=torch.from_numpy(target.astype(np.float32)),
                    d=torch.tensor(s["d"]) if s["d"] is not None else torch.zeros(2), d_ok=torch.tensor(s["d"] is not None),
                    kind=["expert", "perturbed", "negative", "carry"].index(ep["kind"]), phase=torch.tensor(s["phase"]))


class Residual(nn.Module):
    """V10's comparator as the experience reader + a per-step residual head on pi0.5's chunk. bottleneck: the head
    gets no image features (h): its context is the reference-relative d estimate and the state only (the hand offset
    from the reference, grippers, progress), so the current frames reach Delta only through d (and the scalar gate) --
    a correction that ignores the reference cannot be expressed."""

    def __init__(self, backbone, w=256, heads=8, layers=2, h_dim=2048, bottleneck=False, state_dim=9):
        super().__init__()
        self.bb, self.bottleneck = backbone, bool(bottleneck)
        self.ctx = nn.Sequential(nn.Linear((state_dim if bottleneck else h_dim) + 2, w), nn.GELU(), nn.Linear(w, w))
        self.step_in = nn.Linear(14, w)
        self.step_pos = nn.Parameter(torch.randn(1, CHUNK, w) * 0.02)
        self.blocks = nn.ModuleList([T.Block(w, heads, cross=False) for _ in range(layers)])
        self.norm = nn.LayerNorm(w)
        self.head = nn.Linear(w, 7)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # Delta = 0 at the start: the branch off = pi0.5
        self.gate = nn.Linear(h_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 2.0)

    def forward(self, feats, state, lang, drop, d_scale, a_pi, a_ref):
        d, _, _, h = self.bb(feats, state, lang, drop, d_scale, return_h=True)
        h = h.float()
        c = self.ctx(torch.cat([state.float() if self.bottleneck else h, 100 * d.float()], 1))
        x = torch.cat([c[:, None], self.step_in(torch.cat([a_pi, a_ref], -1)) + self.step_pos], 1)
        for blk in self.blocks:
            x = blk(x)
        g = torch.sigmoid(self.gate(h))
        return g[:, :, None] * self.head(self.norm(x[:, 1:])), d, g[:, 0]


def weights(args, device):
    w = torch.ones(CHUNK, device=device)
    w[:EXEC] = args.w_exec
    return w / w.sum()


@torch.no_grad()
def evaluate(model, loaders, args):
    model.eval()
    out = {}
    for name, loader in loaders.items():
        rec = collections.defaultdict(list)
        for b in loader:
            b = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                delta, d, g = model(b["feats"], b["state"], b["lang"], b["drop"], args.d_scale, b["a_pi"], b["a_ref"])
            rms = lambda x: ((x[:, :EXEC] ** 2).mean((1, 2))).sqrt()
            rec["model"].append(rms(delta.float() - b["target"]).cpu())
            rec["gate"].append(g.float().cpu())
            rec["kind"].append(b["kind"].cpu())
            if name == "true":
                rec["zero"].append(rms(b["target"]).cpu())
                rec["copy"].append(rms(b["a_ref"] - b["a_pi"] - b["target"]).cpu())
                err = (d.float() - b["d"]).norm(dim=1)
                rec["d_err"].append(torch.where(b["d_ok"], err, torch.full_like(err, float("nan"))).cpu())
        r = {k: torch.cat(v).numpy() for k, v in rec.items()}
        for kind, nm in ((0, "expert"), (2, "negative")):
            sel = r["kind"] == kind
            if sel.any():
                out[f"{name}_{nm}_rms"] = round(float(r["model"][sel].mean()), 4)
                out[f"{name}_{nm}_gate"] = round(float(r["gate"][sel].mean()), 3)
                if name == "true":
                    out[f"{nm}_zero_rms"] = round(float(r["zero"][sel].mean()), 4)
                    out[f"{nm}_copy_rms"] = round(float(r["copy"][sel].mean()), 4)
        if name == "true":
            out["d_err_cm_median"] = round(float(100 * np.nanmedian(r["d_err"])), 3)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats", default=f"{V}/feats")
    ap.add_argument("--api", default=f"{V11}/api")
    ap.add_argument("--label", choices=("clear", "astar"), default="clear",
                    help="expert-pair label: clear = a_pi without the camera move - a_pi (v11_labels.py); astar = a* - a_pi")
    ap.add_argument("--labels", default=f"{V11}/labels_cam", help="--label clear: v11_labels.py output")
    ap.add_argument("--init", default=f"{V}/train/run1/best.pt", help="V10 comparator to start the backbone from")
    ap.add_argument("--freeze_bb", action="store_true", help="keep the V10 backbone fixed (no d loss)")
    ap.add_argument("--bottleneck", action="store_true",
                    help="the residual head sees d and the state only, no image features (Residual docstring)")
    ap.add_argument("--bb_lr_mult", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--w_exec", type=float, default=3.0)
    ap.add_argument("--w_d", type=float, default=0.1)
    ap.add_argument("--drop_agent", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    ck = torch.load(args.init, weights_only=False)
    a10 = ck["args"]
    args.d_scale, args.width, args.pool = a10["d_scale"], a10["width"], a10["pool"]
    args.kinds, args.dlabel, args.state = ["expert", "negative"], "true", "abs"
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    idx = T.build_index(args)
    from pace.pi05 import InstructionText

    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(idx["lang"].values()))
    lang_vec = {t: text.embed(l).float() for t, l in idx["lang"].items()}
    tr = V11Data(idx, "train", args, lang_vec, True)
    va = {m: V11Data(idx, "val", args, lang_vec, False, ref_mode=m) for m in ("true", "other", "none")}
    print(f"[data] train {len(tr)} samples, val {len(va['true'])} (x 3 reference modes)", flush=True)
    kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=True)
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, drop_last=True, **kw)
    vls = {m: torch.utils.data.DataLoader(v, 64, shuffle=False, **kw) for m, v in va.items()}
    D = np.load(f"{args.feats}/pca.npz")["comp"].shape[0]
    bb = T.Comparator(D, w=args.width, pool=args.pool)
    bb.load_state_dict(ck["model"])
    model = Residual(bb, bottleneck=args.bottleneck).cuda()
    if args.freeze_bb:
        for p in model.bb.parameters():
            p.requires_grad_(False)
    groups = [dict(params=[p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad], lr=args.lr)]
    if not args.freeze_bb:
        groups.append(dict(params=list(model.bb.parameters()), lr=args.lr * args.bb_lr_mult))
    print(f"[model] {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters "
          f"({sum(p.numel() for g in groups for p in g['params']) / 1e6:.2f} M trained)", flush=True)
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)
    total = args.max_steps or args.epochs * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    W = weights(args, "cuda")
    ev = evaluate(model, vls, args)  # step 0: the branch is off (Delta = 0)
    print(f"[val] step 0: {json.dumps(ev)}", flush=True)
    step, best, t0 = 0, float("inf"), time.time()
    log = open(f"{args.out}/log.jsonl", "a")
    log.write(json.dumps(dict(step=0, **ev)) + "\n")
    run = collections.defaultdict(float)
    while step < total:
        for b in tl:
            b = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                delta, d, g = model(b["feats"], b["state"], b["lang"], b["drop"], args.d_scale, b["a_pi"], b["a_ref"])
            l_res = (((delta.float() - b["target"]) ** 2).mean(-1) * W).sum(-1).mean()
            dm = b["d_ok"].float()
            l_d = ((torch.nn.functional.smooth_l1_loss(100 * d.float(), 100 * b["d"], beta=0.5, reduction="none").sum(1) * dm).sum()
                   / dm.sum().clamp(min=1))
            loss = l_res + (0.0 if args.freeze_bb else args.w_d) * l_d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in (("loss", loss), ("res", l_res), ("d_cm", l_d), ("gate", g.mean())):
                run[k] += float(v.detach())
            if step % 100 == 0:
                print(f"[train] step {step}/{total} ({time.time() - t0:.0f}s) " +
                      " ".join(f"{k} {v / 100:.4f}" for k, v in run.items()), flush=True)
                run.clear()
            if step % args.eval_every == 0 or step == total:
                ev = evaluate(model, vls, args)
                log.write(json.dumps(dict(step=step, **ev)) + "\n")
                log.flush()
                print(f"[val] step {step}: {json.dumps(ev)}", flush=True)
                torch.save(dict(model=model.state_dict(), args=vars(args), step=step, val=ev), f"{args.out}/last.pt")
                if ev["true_expert_rms"] < best:
                    best = ev["true_expert_rms"]
                    torch.save(dict(model=model.state_dict(), args=vars(args), step=step, val=ev), f"{args.out}/best.pt")
            if step >= total:
                break
    print(f"-> {args.out} (best validation expert rms {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
