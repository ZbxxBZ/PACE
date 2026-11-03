"""
V11 + uncertainty (10-30, scheme B; user: "可以做 b"): the V12 residual head's idea moved to the LeRobot V11 -- the model
also states how wrong its correction may be, and the closed loop shrinks the correction where it is unsure, so V11
leaves pi0.5 alone where it has nothing reliable to add (target: no harm on the non-camera variants).
The trained V11 (train_v11_v9.py, run2_4090 best.pt) stays frozen: branch, residual head, pi0.5 -- the correction Delta
validated in the closed loop is unchanged. A new small head reads the same input as the residual head (the branch's
effect on the action tokens, (h_on - h_off) / rms(h_off)) and outputs a log-variance per executed step and action dim.
Loss: Gaussian NLL of the (detached) error Delta - Delta*, first 10 steps only. Everything but the new head runs under
no_grad (forward only).
Execution (closed loop): Delta * f, f = signal / (signal + var) per element (Wiener). The rule is chosen ONCE after
training on the validation set only (mode true, all kinds incl. negatives, executed 10 steps; criterion: lowest mean
RMS of Delta * f - Delta*), among signal = Delta^2 and signal = constant s per action dim (s = c x the label's mean
square on the validation true set, c in C_GRID), and stored in the checkpoint ("rule"); never tuned on the test cells.
Validation (same fixed samples as training, modes true / other): calibration (share of |error| within 1 / 2 sigma; a
Gaussian gives 0.68 / 0.95), RMS of Delta vs Delta * f against the label per kind, and the correction size on the
negatives and with an other-task reference (should shrink most).
For closed_loop_v11.py (the peer session runs the closed loop): load_var_head(path, device) -> (head, rule),
var(head, h_on, h_off) -> variance [B, 50, 7] (normalized action units, same shape as Delta), shrink(delta, variance,
rule) -> the executed correction.
    python train_v11_var.py --ckpt runs/v11/train/v9_run2_4090/best.pt --out runs/v11/train/v9_var
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

import train_v10b as T
import train_v11_v9 as V

EXEC = V.EXEC


class VarHead(nn.Module):
    """log-variance [B, 50, 7] from the normalized branch effect; starts at the label's typical size."""

    def __init__(self, dim=1024, hidden=256, init_std=0.25):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 7)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, math.log(init_std ** 2))

    def forward(self, diff):
        return self.fc2(F.gelu(self.fc1(diff)))


class Tap:
    """Keeps the residual head's input (the normalized diff) of its latest call."""

    def __init__(self, head):
        self.diff = None
        orig = head.forward

        def fwd(h_on, h_off):
            self.diff = normalized_diff(h_on, h_off)
            return orig(h_on, h_off)

        head.forward = fwd


C_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)


def normalized_diff(h_on, h_off):
    """The residual head's input (train_v11_v9.DiffHead): the branch's effect relative to the size of h_off."""
    return (h_on - h_off) / (h_off.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()


def shrink(delta, variance, rule):
    """Wiener shrinkage of the correction; rule = {"signal": "delta2"} or {"signal": "const", "s": [7 floats]}."""
    if rule["signal"] == "delta2":
        sig = delta.pow(2)
    else:
        sig = torch.tensor(rule["s"], device=delta.device, dtype=delta.dtype).expand_as(delta)
    return delta * sig / (sig + variance)


def load_var_head(path, device="cuda"):
    """-> (VarHead in eval mode, the shrink rule chosen on validation)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    vh = VarHead().to(device)
    vh.load_state_dict(ck["var_head"])
    vh.eval()
    return vh, ck["rule"]


@torch.no_grad()
def var(head, h_on, h_off):
    """Variance [B, 50, 7] of the correction's error (normalized action units, same shape as Delta)."""
    return head(normalized_diff(h_on.float(), h_off.float())).exp()


@torch.no_grad()
def collect(run, tap, vh, loaders):
    """Per mode: executed-part Delta, variance, label [N, 10, 7] and kind [N] of the validation samples."""
    vh.eval()
    out = {}
    for mode, loader in loaders.items():
        rec = collections.defaultdict(list)
        for b in loader:
            tap.diff = None  # Runner returns zeros without calling the head when no row has a reference
            delta, _, _ = run(b)
            tgt = b["target"].to(run.device)
            if tap.diff is None or delta.abs().sum() == 0:
                variance = torch.full_like(delta, 0.25 ** 2)
            else:
                variance = vh(tap.diff.float()).exp()
            for k, v in (("d", delta), ("v", variance), ("y", tgt)):
                rec[k].append(v[:, :EXEC].float().cpu())
            rec["kind"].append(b["kind"])
        out[mode] = {k: torch.cat(v) for k, v in rec.items()}
    vh.train()
    return out


def report(col, rule):
    """Calibration and errors per mode / kind (mean over samples of the per-sample RMS over 10 steps x 7 dims)."""
    rms = lambda x: x.pow(2).mean((1, 2)).sqrt()
    out = {}
    for mode, c in col.items():
        d, v, y = c["d"], c["v"], c["y"]
        s = shrink(d, v, rule)
        z = (d - y).abs() / v.sqrt()
        m = dict(zero=rms(y), raw=rms(d - y), shrunk=rms(s - y), size_raw=rms(d), size_shrunk=rms(s),
                 in1=(z < 1).float().mean((1, 2)), in2=(z < 2).float().mean((1, 2)), sigma=v.sqrt().mean((1, 2)))
        for kind, nm in enumerate(V.KINDS):
            sel = c["kind"] == kind
            if sel.any():
                out[f"{mode}_{nm}"] = {k: round(float(x[sel].mean()), 4) for k, x in m.items()}
    return out


def choose_rule(col):
    """The shrink rule with the lowest mean per-sample RMS (Delta * f - label) on validation mode true, all kinds."""
    c = col["true"]
    ms = c["y"].pow(2).mean((0, 1))  # label mean square per action dim
    cands = [{"signal": "delta2"}] + [{"signal": "const", "s": (cc * ms).tolist(), "c": cc} for cc in C_GRID]
    scores = []
    for r in cands:
        e = (shrink(c["d"], c["v"], r) - c["y"]).pow(2).mean((1, 2)).sqrt().mean()
        scores.append(float(e))
    raw = float((c["d"] - c["y"]).pow(2).mean((1, 2)).sqrt().mean())
    best = cands[int(np.argmin(scores))]
    return dict(best, val_score=round(min(scores), 4), val_raw=round(raw, 4),
                candidates=[[json.dumps({k: v for k, v in r.items() if k != "s"}), round(s, 4)] for r, s in zip(cands, scores)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{V.V11}/train/v9_run2_4090/best.pt")
    ap.add_argument("--out", default=f"{V.V11}/train/v9_var")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_n", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--smoke", action="store_true")
    cli = ap.parse_args()
    ck = torch.load(cli.ckpt, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ck["args"])
    args.bs, args.val_n = cli.bs, cli.val_n
    if cli.smoke:
        cli.steps, args.val_n, cli.eval_every = 20, 16, 20
    os.makedirs(cli.out, exist_ok=True)
    torch.manual_seed(0)
    kinds = list(args.kinds)
    args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
    idx = T.build_index(args)
    args.kinds = kinds
    if "pi05_aug" in kinds:
        idx = V.add_pi05_aug(idx)
    tr = V.V11Data(idx, "train", args, True)
    va = {m: V.V11Data(idx, "val", args, False, mode=m, limit=args.val_n) for m in ("true", "other")}
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, drop_last=True, num_workers=cli.workers,
                                     persistent_workers=True)
    vls = {m: torch.utils.data.DataLoader(v, args.bs, shuffle=False, num_workers=2) for m, v in va.items()}
    run = V.Runner(args)
    run.adapter.load_state_dict(ck["adapter"])
    run.head.load_state_dict(ck["head"])
    run.adapter.eval()
    for p in list(run.adapter.parameters()) + list(run.head.parameters()):
        p.requires_grad_(False)
    tap = Tap(run.head)
    vh = VarHead().to(run.device)
    opt = torch.optim.AdamW(vh.parameters(), lr=cli.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / cli.steps)))
    json.dump(dict(vars(cli), base_args=vars(args)), open(f"{cli.out}/args.json", "w"), indent=1)
    log = open(f"{cli.out}/log.jsonl", "a")
    print(f"[data] train {len(tr)}, val {len(va['true'])} per mode; base ckpt step {ck.get('step')}", flush=True)
    D2 = {"signal": "delta2"}  # the rule reported during training; the executed rule is chosen once at the end
    col = collect(run, tap, vh, vls)
    ev = report(col, D2)
    log.write(json.dumps(dict(step=0, **ev)) + "\n")
    log.flush()
    print(f"[val] step 0: {json.dumps(ev)}", flush=True)
    step, t0, it, acc = 0, time.time(), iter(tl), 0.0
    while step < cli.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(cli.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(tl)
                b = next(it)
            tap.diff = None
            with torch.no_grad():
                delta, _, _ = run(b)
                diff = None if tap.diff is None else tap.diff.float()
            if diff is None or delta.abs().sum() == 0:
                continue
            tgt = b["target"].to(run.device)
            lv = vh(diff)[:, :EXEC]
            e2 = (delta[:, :EXEC] - tgt[:, :EXEC]).pow(2)  # detached: the mean is frozen
            loss = 0.5 * (e2 / lv.exp() + lv).mean()
            (loss / cli.accum).backward()
            acc += float(loss) / cli.accum
        torch.nn.utils.clip_grad_norm_(vh.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0 or cli.smoke:
            n = 1 if cli.smoke else 50
            print(f"[train] step {step}/{cli.steps} ({time.time() - t0:.0f}s) nll {acc / n:.4f}", flush=True)
            acc = 0.0
        if step % cli.eval_every == 0 or step == cli.steps:
            col = collect(run, tap, vh, vls)
            ev = report(col, D2)
            log.write(json.dumps(dict(step=step, **ev)) + "\n")
            log.flush()
            print(f"[val] step {step}: {json.dumps(ev)}", flush=True)
            torch.save(dict(var_head=vh.state_dict(), base_ckpt=cli.ckpt, step=step, val=ev, rule=D2), f"{cli.out}/last.pt")
    rule = choose_rule(col)  # validation only, once, after training; the closed loop uses it unchanged
    final = report(col, rule)
    torch.save(dict(var_head=vh.state_dict(), base_ckpt=cli.ckpt, step=step, val=final, rule=rule), f"{cli.out}/final.pt")
    json.dump(dict(rule=rule, val=final), open(f"{cli.out}/final_val.json", "w"), indent=1)
    print(f"[rule] {json.dumps(rule)}", flush=True)
    print(f"[final] {json.dumps(final)}", flush=True)
    print(f"-> {cli.out}/final.pt", flush=True)


if __name__ == "__main__":
    main()
