"""
V11 with V9's structure (10-27; user: V11 is derived from V9; desktop design doc 10.14c). pi0.5 stays frozen. The
experience branch is V9's PACE adapter unchanged (pace.core.ExperienceAdapter as V9b trained it, docs/V9_training.md
4.5: v2 reference encoder with reference isolation, gated cross-attention after action-expert layers 1, 3, ..., 17,
input injection of the reference chunk, relevance gate, no generic branch). What changes is how its effect is read and
what it is trained on:
  1. pi0.5 samples its chunk a_pi as usual, branch off (precomputed: v11_check.py --dump -> runs/v11/api).
  2. The action expert runs on the clean chunk (x = a_pi, flow time --read_time) twice: branch off -> hidden states
     h_off of the 50 action tokens (the input of pi0.5's action_out_proj), branch on (reference fed) -> h_on.
  3. A new head reads their difference, relative to the size of h_off: Delta = W2 gelu(W1 (h_on - h_off) / rms(h_off)),
     bias-free, so Delta = 0 exactly whenever the branch adds nothing (no reference, or its relevance gate closed): a
     correction that does not come through a reference cannot be expressed. (The branch could still use a reference as
     a mere trigger -- the other-task / unaligned controls and the d head watch for that.) W2 starts small, not zero:
     the branch's own output projections start at zero, and two zero factors would never move.
  4. An aux head on the same difference predicts d (this phase's reference-relative shift; true d from the simulator),
     so what the branch injects must carry the comparison of the current scene with the reference.
  a = a_pi + Delta (normalized action space).
Data: V10's pairs on the 10 original libero_10 tasks (init states 6-44 train, 45-49 validation; scene augmentation incl.
agentview camera moves). Successful expert pairs, --label clear (default, 10-27 after the label pilot): Delta* =
a_pi(the same state seen without the camera move) - a_pi, same noise seed (v11_labels.py) -- only what the camera move
changes; --label astar (first version): Delta* = normalize(a*) - a_pi, a* = the executed reference + true-d controller
(mostly pi0.5's own variety, the residual split of 10-27). Negative pairs: Delta* = 0 (failed expert replays are left
out). With the clear label any rendered state can be labelled, so --kinds (default all) adds V10's perturbed pairs
(hand pushed off 1-3 cm), carry pairs (a phase entered with the previous phase's d) and pi0.5's own rollouts under the
augmentations (pi05_aug, failures included; one same-task reference per episode, no d label; add_pi05_aug). All of it
is the ORIGINAL libero_10 scenes (LIBERO-plus's copies of the 10 original tasks) with our render-only augmentation; no
LIBERO-plus variant is used for training. One reference
per sample (the pair's), aligned by time (j = t), given to the adapter as V9 gives one: the 50-step chunk from j, the
classic scalars (pace.core.SCALAR_NAMES; similarity = mean cosine of the current and the reference pi0.5 grid cells,
align cost 0), success, image tokens and state at j (v11_check.py --refvis).
Loss: squared error of Delta per step (first 10 steps x --w_exec) + --w_d x smooth-L1 of d in cm (expert samples).
Validation (fixed subset of init states 45-49) per reference mode -- true; other (a reference of another task, same
step); unaligned (the true reference read at a random step); none (Delta = 0 by construction, i.e. pi0.5) -- RMS of
the executed part, against Delta = 0 (pi0.5 alone) and Delta = a_ref - a_pi (copy the reference); d error.
    python train_v11_v9.py --out runs/v11/train/v9_run1
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
from v11_check import load_pi05, raw_obs

V, V11 = "/root/pace/runs/v10b", "/root/pace/runs/v11"
CALL, CHUNK, EXEC = 10, 50, 10
MODES = ("true", "other", "unaligned", "none")
FRAME_KEYS = ("agent", "wrist", "eef_pos", "eef_quat", "gripper", "joints")
KINDS = ("expert", "perturbed", "negative", "carry", "pi05_aug")
VAL_INITS = range(45, 50)


def add_pi05_aug(idx, seed=5):
    """pi0.5's own rollouts under the scene augmentations (runs/v10b/pi05_aug, failures included) as extra samples: each
    episode gets one reference, a successful pi0.5 reference of the same task at another init state of the same split
    (fixed per episode), as the explore data did; no d label (no pair: the shift is undefined there)."""
    refs_of = collections.defaultdict(list)
    for k in sorted(idx["refs"]):
        tid, init = int(k.split("_i")[0][1:]), int(k.split("_i")[1])
        refs_of[(tid, init in VAL_INITS)].append((init, k))
    n0 = len(idx["eps"])
    for r in map(json.loads, open(f"{V}/pi05_aug/episodes.jsonl")):
        key = f"t{r['task_id']}_i{r['init']}_a{r['aug_idx']}"
        path = f"{V}/frames/pi05_aug/{key}_a0.npz"
        if not os.path.exists(path):
            continue
        val = r["init"] in VAL_INITS
        pool = [k for init, k in refs_of[(r["task_id"], val)] if init != r["init"]]
        if not pool:
            continue
        rng = np.random.default_rng([r["task_id"], r["init"], r["aug_idx"], seed])
        e = len(idx["eps"])
        idx["eps"].append(dict(kind="pi05_aug", key=key, fkey=f"{key}_a0", ref=pool[int(rng.integers(len(pool)))],
                               split="val" if val else "train", task=r["task_id"], meta=T.frame_meta(path)))
        for c, t in enumerate(idx["eps"][e]["meta"]["t"]):
            idx["samples"].append(dict(ep=e, c=c, t=int(t), d=None, phase=-1))
    print(f"[index] + {len(idx['eps']) - n0} pi05_aug episodes", flush=True)
    return idx


def chunk(actions, t):
    """Actions t .. t + CHUNK - 1 of an episode; past the end: hold still with the last gripper command."""
    n = len(actions)
    return np.stack([actions[t + k] if t + k < n else np.r_[np.zeros(6), actions[-1][6]] for k in range(CHUNK)])


class V11Data(torch.utils.data.Dataset):
    """V10's pair samples at pi0.5 calls with a_pi, the residual label and one reference in V9's input format."""

    def __init__(self, idx, split, args, train, mode="true", limit=0):
        self.idx, self.args, self.train, self.mode = idx, args, train, mode
        z = np.load(f"{args.api}/norm.npz")
        self.mean, self.std = z["mean"], z["std"]
        refs_rows = [json.loads(l) for l in open(f"{V}/refs/episodes.jsonl")]
        self.ref_file = {f"t{r['task_id']}_i{r['init']}": f"{V}/refs/{r['actions']}" for r in refs_rows}
        self.ref_task = {f"t{r['task_id']}_i{r['init']}": r["task_id"] for r in refs_rows}
        self.lang = {r["task_id"]: r["language"] for r in refs_rows}
        have = {k for k in idx["refs"] if os.path.exists(f"{args.refvis}/{k}_a0.npz")}
        self.items = []
        for i, s in enumerate(idx["samples"]):
            ep = idx["eps"][s["ep"]]
            if ep["kind"] == "expert" and not ep["meta"]["success"]:
                continue  # a failed expert replay: its executed actions are not a correct answer
            if ep["kind"] not in args.kinds or (args.label == "astar" and ep["kind"] not in ("expert", "negative")):
                continue  # a* exists for the expert pairs only
            if (ep["split"] == split and s["t"] % CALL == 0 and ep["ref"] in have
                    and os.path.exists(f"{args.api}/{ep['kind']}/{ep['fkey']}.npy")
                    and (ep["kind"] == "negative" or args.label != "clear"
                         or os.path.exists(f"{args.labels}/{ep['kind']}/{ep['fkey']}.npy"))):
                self.items.append(i)
        if limit and len(self.items) > limit:
            self.items = sorted(np.random.default_rng(0).choice(self.items, limit, replace=False).tolist())
        self.others = collections.defaultdict(list)  # task -> references of the other tasks
        for k in sorted(have):
            for t_ in self.lang:
                if self.ref_task[k] != t_:
                    self.others[t_].append(k)
        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _small(self, path, field=None):
        """Small arrays cached per worker as plain arrays (no open file kept: mmaps / npz handles hold a descriptor)."""
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

    def __getitem__(self, i):
        s = self.idx["samples"][self.items[i]]
        ep = self.idx["eps"][s["ep"]]
        rng = np.random.default_rng() if self.train else np.random.default_rng(self.items[i])
        rkey = ep["ref"]
        if self.mode == "other":
            pool = self.others[ep["task"]]
            rkey = pool[int(rng.integers(len(pool)))]
        rv = self._small(f"{self.args.refvis}/{rkey}_a0.npz")
        t = s["t"]
        want = int(rng.integers(0, int(rv["t"][-1]) + 1)) if self.mode == "unaligned" else t
        calls = np.where(rv["t"] % CALL == 0)[0]  # the reference's own pi0.5 calls
        j = int(calls[np.argmin(np.abs(rv["t"][calls] - want))])
        ref_t = int(rv["t"][j])
        with np.load(f"{V}/frames/{ep['kind']}/{ep['fkey']}.npz") as fr:
            cur = {k: np.array(fr[k][s["c"]]) for k in FRAME_KEYS}
        a_pi = np.asarray(self._small(f"{self.args.api}/{ep['kind']}/{ep['fkey']}.npy")[s["c"]], np.float32)
        if ep["kind"] != "negative" and self.args.label == "clear":  # what pi0.5 would do seeing clearly (same seed)
            target = np.asarray(self._small(f"{self.args.labels}/{ep['kind']}/{ep['fkey']}.npy")[s["c"]], np.float32) - a_pi
        elif ep["kind"] == "expert":
            target = self.norm(chunk(self._small(f"{V}/pairs/expert/{ep['key']}.npz", "actions").astype(np.float64), t)) - a_pi
        else:  # negative: this reference should not be followed -> leave pi0.5 alone
            target = np.zeros_like(a_pi)
        a_ref = self.norm(chunk(self._small(self.ref_file[rkey]).astype(np.float64), ref_t))
        L = len(calls)
        jc = int(np.searchsorted(calls, j))
        return dict(**cur, lang=self.lang[ep["task"]], t=t, a_pi=torch.from_numpy(a_pi),
                    target=torch.from_numpy(target.astype(np.float32)), a_ref=torch.from_numpy(a_ref),
                    ref_vis=torch.from_numpy(np.asarray(rv["vis"][j])), ref_state=torch.from_numpy(rv["state"][j].astype(np.float32)),
                    ref_pos=torch.tensor([jc / max(L - 1, 1), jc, L], dtype=torch.float32),
                    valid=torch.tensor(self.mode != "none"),
                    d=torch.tensor(s["d"]) if s["d"] is not None else torch.zeros(2), d_ok=torch.tensor(s["d"] is not None),
                    kind=KINDS.index(ep["kind"]))


class DiffHead(nn.Module):
    """Delta and d from the branch's effect on the action tokens (h_on - h_off, relative to |h_off|); bias-free."""

    def __init__(self, dim=1024, hidden=512, action_dim=7, init_std=1e-3, d_scale=0.08):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, action_dim, bias=False)
        self.fd = nn.Linear(dim, 2, bias=False)
        nn.init.normal_(self.fc2.weight, std=init_std)
        nn.init.normal_(self.fd.weight, std=init_std)
        self.d_scale = d_scale

    def forward(self, h_on, h_off):
        diff = (h_on - h_off) / (h_off.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
        return self.fc2(F.gelu(self.fc1(diff))), self.d_scale * torch.tanh(self.fd(diff.mean(1)))


class Capture:
    """The input of pi0.5's action_out_proj (the action tokens' last hidden states) at its latest call."""

    def __init__(self, module):
        self.h = None
        module.register_forward_pre_hook(self)

    def __call__(self, module, args):
        self.h = args[0]


class Runner:
    def __init__(self, args):
        from lerobot.envs.utils import preprocess_observation
        from pace.core import CALL_SCALE
        from pace.pi05 import PI05Experience, batch_state, build_adapter, prefix_cache, vis_tokens

        self.args, self.pre_obs, self.batch_state, self.vis_tokens, self.prefix_cache = (
            args, preprocess_observation, batch_state, vis_tokens, prefix_cache)
        self.call_scale = CALL_SCALE
        self.policy, _, self.pre, self.env_pre, _, _ = load_pi05()
        for p in self.policy.parameters():
            p.requires_grad_(False)
        self.adapter = build_adapter(self.policy, inject_layers=("every2",), max_refs=args.max_refs, input_inject=True,
                                     position="post_layer", time_cond=False, encoder="v2",
                                     feature_shapes=((32, 2048), 8), enc_dim=512, enc_layers=2, enc_heads=8,
                                     act_segments=10, retrieval="classic", generic="none", relevance="gate",
                                     ref_isolation=True)
        self.pace = PI05Experience(self.policy, self.adapter, runtime_kwargs={"retrieval": "classic"})
        self.pace.attach()
        self.device = self.pace.device
        self.head = DiffHead().to(self.device)
        self.cap = Capture(self.policy.model.action_out_proj)

    def hidden(self, prefix, x, tm):
        self.policy.model.denoise_step(prefix_pad_masks=prefix[0], past_key_values=prefix[1], x_t=x, timestep=tm)
        return self.cap.h.float()

    def __call__(self, b):
        """-> Delta [B, 50, 7], d [B, 2] (both 0 for rows without a reference), relevance stats."""
        dev, B = self.device, len(b["lang"])
        fr = {k: b[k].numpy() for k in FRAME_KEYS}
        obs = self.pre_obs(raw_obs([(fr, i) for i in range(B)]))
        obs["task"] = list(b["lang"])
        obs = self.pre(self.env_pre(obs))
        with torch.no_grad():
            images, img_masks = self.policy._preprocess_images(obs)
            cur_vis = self.vis_tokens(self.policy.model._embed_images(images, img_masks), (0, 1)).float()
            cur_state = self.batch_state(obs)
            prefix = self.prefix_cache(self.policy, obs)
        ref_vis = b["ref_vis"].to(dev).float()
        rp = b["ref_pos"].to(dev)
        sim = (cur_vis * ref_vis).sum(-1).mean(-1)
        call = b["t"].to(dev).float() / CALL
        scalars = torch.stack([rp[:, 0], sim, torch.zeros_like(sim), call / self.call_scale, rp[:, 1] / self.call_scale,
                               rp[:, 2] / self.call_scale], -1)[:, None]
        exp = dict(actions=b["a_ref"].to(dev)[:, None], scalars=scalars, success=torch.ones(B, 1, dtype=torch.long, device=dev),
                   valid=b["valid"].to(dev)[:, None], ref_vis=ref_vis[:, None], ref_state=b["ref_state"].to(dev)[:, None],
                   cur_vis=cur_vis, cur_state=cur_state)
        x = F.pad(b["a_pi"].to(dev), (0, self.policy.config.max_action_dim - b["a_pi"].shape[-1]))
        tm = torch.full((B,), self.args.read_time, device=dev)
        self.pace.set_exp(None)
        with torch.no_grad():
            h_off = self.hidden(prefix, x, tm)
        self.adapter.relevance_stats = {}
        self.pace.set_exp(exp)
        if self.pace.encoded is None:  # no row has a reference: the branch adds nothing
            self.pace.set_exp(None)
            return torch.zeros(B, CHUNK, 7, device=dev), torch.zeros(B, 2, device=dev), {}
        h_on = self.hidden(prefix, x, tm)
        stats = dict(self.adapter.relevance_stats)
        self.pace.set_exp(None)
        delta, d = self.head(h_on, h_off)
        return delta, d, stats

    def params(self):
        return [p for p in self.adapter.parameters() if p.requires_grad] + list(self.head.parameters())

    def save(self, path, **extra):
        torch.save(dict(adapter=self.adapter.state_dict(), config=self.adapter.config, head=self.head.state_dict(),
                        args=vars(self.args), **extra), path)


def weights(args, device):
    w = torch.ones(CHUNK, device=device)
    w[:EXEC] = args.w_exec
    return w / w.sum()


@torch.no_grad()
def evaluate(run, loaders, args):
    run.adapter.eval()
    out = {}
    rms = lambda x: (x[:, :EXEC] ** 2).mean((1, 2)).sqrt()
    for mode, loader in loaders.items():
        rec, rel = collections.defaultdict(list), collections.defaultdict(list)
        for b in loader:
            delta, d, stats = run(b)
            tgt = b["target"].to(run.device)
            rec["model"].append(rms(delta - tgt).cpu())
            rec["kind"].append(b["kind"])
            for k, v in stats.items():
                rel[k].append(v)
            if mode == "true":
                rec["zero"].append(rms(tgt).cpu())
                rec["copy"].append(rms(b["a_ref"].to(run.device) - b["a_pi"].to(run.device) - tgt).cpu())
                err = (d - b["d"].to(run.device)).norm(dim=1)
                rec["d_err"].append(torch.where(b["d_ok"].to(run.device), err, torch.full_like(err, float("nan"))).cpu())
                rec["d_zero"].append(torch.where(b["d_ok"].to(run.device), b["d"].to(run.device).norm(dim=1),
                                                 torch.full_like(err, float("nan"))).cpu())
        r = {k: torch.cat(v).numpy() for k, v in rec.items()}
        for kind, nm in enumerate(KINDS):
            sel = r["kind"] == kind
            if sel.any():
                out[f"{mode}_{nm}_rms"] = round(float(r["model"][sel].mean()), 4)
                if mode == "true":
                    out[f"{nm}_zero_rms"] = round(float(r["zero"][sel].mean()), 4)
                    out[f"{nm}_copy_rms"] = round(float(r["copy"][sel].mean()), 4)
        for k, v in rel.items():
            out[f"{mode}_{k}"] = round(float(np.mean(v)), 3)
        if mode == "true":
            out["d_err_cm_median"] = round(float(100 * np.nanmedian(r["d_err"])), 3)
            out["d_zero_cm_median"] = round(float(100 * np.nanmedian(r["d_zero"])), 3)
    run.adapter.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--feats", default=f"{V}/feats", help="only for V10's sample index (train_v10b.build_index)")
    ap.add_argument("--api", default=f"{V11}/api")
    ap.add_argument("--refvis", default=f"{V11}/refvis")
    ap.add_argument("--label", choices=("clear", "astar"), default="clear",
                    help="expert-pair label: clear = a_pi seeing the scene without the camera move - a_pi (v11_labels.py; "
                         "decided 10-27 after the label pilot); astar = normalize(a*) - a_pi (the first version)")
    ap.add_argument("--labels", default=f"{V11}/labels_cam", help="--label clear: v11_labels.py output")
    ap.add_argument("--kinds", nargs="+", default=list(KINDS),
                    help="sample kinds (10-27, user: + perturbed, carry, pi05_aug; these need --label clear)")
    ap.add_argument("--max_refs", type=int, default=4, help="adapter slots (one is used per sample)")
    ap.add_argument("--read_time", type=float, default=0.05, help="flow time of the two passes on the clean chunk")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=8, help="samples per forward")
    ap.add_argument("--accum", type=int, default=4, help="forwards per optimizer step (32 samples per step)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--w_exec", type=float, default=3.0)
    ap.add_argument("--w_d", type=float, default=0.1)
    ap.add_argument("--val_n", type=int, default=400, help="validation samples per reference mode")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="20 steps, 16 validation samples per mode")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.val_n, args.eval_every, args.warmup = 20, 16, 20, 5
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    kinds = list(args.kinds)
    args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
    idx = T.build_index(args)  # V10's pair samples (cached next to --out)
    args.kinds = kinds
    if "pi05_aug" in kinds:
        idx = add_pi05_aug(idx)
    tr = V11Data(idx, "train", args, True)
    va = {m: V11Data(idx, "val", args, False, mode=m, limit=args.val_n) for m in MODES}
    print(f"[data] train {len(tr)} samples, validation {len(va['true'])} per mode x {len(MODES)}", flush=True)
    tl = torch.utils.data.DataLoader(tr, args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                                     persistent_workers=True)
    vls = {m: torch.utils.data.DataLoader(v, args.bs, shuffle=False, num_workers=2, persistent_workers=True)
           for m, v in va.items()}
    run = Runner(args)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    params = run.params()
    print(f"[model] adapter {sum(p.numel() for p in run.adapter.parameters()) / 1e6:.1f}M + head "
          f"{sum(p.numel() for p in run.head.parameters()) / 1e6:.2f}M parameters trained", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    W = weights(args, run.device)
    log = open(f"{args.out}/log.jsonl", "a")
    ev = evaluate(run, vls, args)  # step 0: the branch adds nothing yet
    print(f"[val] step 0: {json.dumps(ev)}", flush=True)
    log.write(json.dumps(dict(step=0, **ev)) + "\n")
    log.flush()
    step, best, t0, it = 0, float("inf"), time.time(), iter(tl)
    acc = collections.defaultdict(float)
    run.adapter.train()
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(tl)
                b = next(it)
            delta, d, stats = run(b)
            tgt = b["target"].to(run.device)
            l_res = (((delta - tgt) ** 2).mean(-1) * W).sum(-1).mean()
            dm = b["d_ok"].to(run.device).float()
            l_d = (F.smooth_l1_loss(100 * d, 100 * b["d"].to(run.device), beta=0.5, reduction="none").sum(1) * dm).sum() \
                / dm.sum().clamp(min=1)
            ((l_res + args.w_d * l_d) / args.accum).backward()
            acc["res"] += float(l_res.detach()) / args.accum
            acc["d_cm"] += float(l_d.detach()) / args.accum
            for k, v in stats.items():
                acc[k] += v / args.accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 50 == 0 or args.smoke:
            n = 1 if args.smoke else 50
            print(f"[train] step {step}/{args.steps} ({time.time() - t0:.0f}s) "
                  + " ".join(f"{k} {v / n:.4f}" for k, v in acc.items()), flush=True)
            acc.clear()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(run, vls, args)
            log.write(json.dumps(dict(step=step, **ev)) + "\n")
            log.flush()
            print(f"[val] step {step}: {json.dumps(ev)}", flush=True)
            run.save(f"{args.out}/last.pt", step=step, val=ev)
            if ev.get("true_expert_rms", float("inf")) < best:
                best = ev["true_expert_rms"]
                run.save(f"{args.out}/best.pt", step=step, val=ev)
    print(f"-> {args.out} (best validation expert rms {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
