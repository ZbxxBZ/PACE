"""
Is the reference image path of the V11 branch used, and can it be used at all? (11-01, read only on the trained runs.)
  look <run> <ckpt>: on validation samples, the share of cross-attention on the null token, summary, action segments and image
    tokens per injection layer; how strongly Delta reacts to each reference input (gradient x input norm); the gradient of the
    training loss on the encoder's image, action and summary inputs.
  overfit <on|off> <real|blank|act>: a fresh branch on 96 samples, one shared target pattern T, +T when the reference image comes from
    one half of the tasks and -T from the other, for the same current state; the rest of the reference is the sample's own and the
    similarity scalar is zeroed, so only the image tokens tell the sign. real: cos(Delta, target) should reach about 1 if the path
    carries information; blank (zero images): stays near 0; act: positive control, the sign is in the action chunk, images fixed.
    (First version, one random target per sample: gradients cancel, the branch stayed at 0 in every condition, no information.)
    cmp, cmpblank, cmpcam (11-01): the branch plus CompareHead. cmp as real; cmpblank as blank; cmpcam: the sign is whether the
    reference has its camera in place or moved (same reference, same step), the signal a camera correction would need.
    python v11_imgpath.py overfit off real
"""
import argparse, collections, json, sys
import numpy as np
import torch
import torch.nn.functional as FN
import train_v11_v9 as M

T_ = M.T


class CompareHead(torch.nn.Module):
    """
    Prototype of the explicit comparison: one token per image cell from cur - ref, bias-free, so it is exactly 0 where the two agree and
    cannot serve as a fixed bank of vectors, read by every action step directly, without passing through pi0.5's layers. Output
    zero-initialized and bias-free.
    """

    def __init__(self, vis_dim=2048, d=256, heads=4, tokens=32, dim=1024, action_dim=7):
        super().__init__()
        nn = torch.nn
        self.heads = heads
        self.fc1, self.fc2 = nn.Linear(vis_dim, d, bias=False), nn.Linear(d, d, bias=False)
        self.pos = nn.Parameter(torch.randn(tokens, d) * 0.02)
        self.q = nn.Linear(dim, d)
        self.k, self.v = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        self.o1, self.o2 = nn.Linear(d, d, bias=False), nn.Linear(d, action_dim, bias=False)
        nn.init.zeros_(self.o2.weight)

    def forward(self, cur, ref, h):
        c = self.fc2(FN.gelu(self.fc1((cur - ref) * 32))) * (1 + self.pos)  # cells are unit vectors of 2048 numbers
        B, N, H = h.shape[0], h.shape[1], self.heads
        q = self.q(h.float()).view(B, N, H, -1).transpose(1, 2)
        k = self.k(c).view(B, c.shape[1], H, -1).transpose(1, 2)
        v = self.v(c).view(B, c.shape[1], H, -1).transpose(1, 2)
        out = FN.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1)
        return self.o2(FN.gelu(self.o1(out)))


def setup(run_dir, iso=None):
    args = argparse.Namespace(**json.load(open(f"{run_dir}/args.json")))
    if iso:
        args.ref_isolation = iso
    kinds = list(args.kinds)
    args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
    idx = T_.build_index(args)
    args.kinds = kinds
    return args, M.add_pi05_aug(idx)


def look(run_dir, name):
    args, idx = setup(run_dir)
    ds = M.V11Data(idx, "val", args, False, mode="true", limit=120)
    run = M.Runner(args)
    ck = torch.load(f"{run_dir}/{name}", map_location="cpu", weights_only=False)
    run.adapter.load_state_dict(ck["adapter"]); run.head.load_state_dict(ck["head"]); run.adapter.eval()
    S, rec = dict(on=True), []
    orig = FN.scaled_dot_product_attention

    def sdpa(q, k, v, *a, **kw):
        if S["on"] and k.shape[-2] == 44:
            mask = a[0] if a else kw["attn_mask"]
            w = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5) + mask
            rec.append(w.softmax(-1).mean((1, 2)).detach().float().cpu())
        return orig(q, k, v, *a, **kw)

    FN.scaled_dot_product_attention = sdpa
    W = M.weights(args, run.device)
    sens, grads, n = collections.defaultdict(list), collections.Counter(), 0
    groups = dict(vis_in=["vis_in", "vis_pos"], segment=["segment_mlp", "segment_emb"], summary=["summary_mlp"], context=["context"], xattn=["layers."], inject=["inject_"])
    for b in torch.utils.data.DataLoader(ds, 8, shuffle=True, num_workers=2, drop_last=True, generator=torch.Generator().manual_seed(0)):
        c = dict(b)
        for k in ("ref_vis", "a_ref", "ref_state"):
            c[k] = b[k].float().clone().requires_grad_(True)
        rec.clear() if n else None
        S["on"] = True
        delta = run(c)[0]
        S["on"] = False
        tgt = b["target"].to(run.device)
        loss = (((delta - tgt) ** 2).mean(-1) * W).sum(-1).mean()
        run.adapter.zero_grad(set_to_none=True)
        loss.backward()
        for k in ("ref_vis", "a_ref", "ref_state"):
            g, x = c[k].grad.flatten(1), c[k].detach().flatten(1)
            sens[k].append((g.norm(dim=1) * x.norm(dim=1)).cpu())
        for nm, p in run.adapter.named_parameters():
            if p.grad is not None:
                for g_, keys in groups.items():
                    if any(nm.startswith(x) for x in keys):
                        grads[g_] += float(p.grad.pow(2).sum())
        n += 1
        if n == 1:
            A = torch.stack(rec).mean(1)  # [layers, tokens]
            names = sorted(run.adapter.layers, key=int)
            print("attention share per layer (null / summary / 10 action segments / 32 image tokens):", flush=True)
            for i, l in zip(range(len(A)), names):
                a = A[i]
                print(f"  layer {l:>2s}: {a[0]:.3f} {a[1]:.3f} {a[2:12].sum():.3f} {a[12:].sum():.3f}", flush=True)
            a = A.mean(0)
            print(f"  mean    : null {a[0]:.3f} summary {a[1]:.3f} segments {a[2:12].sum():.3f} images {a[12:].sum():.3f}; per token: segment {a[2:12].mean():.4f} image {a[12:].mean():.4f}", flush=True)
    print("loss sensitivity |dL/dx| * |x| (mean over samples):", {k: round(float(torch.cat(v).mean()), 5) for k, v in sens.items()}, flush=True)
    print("gradient norm of the loss on parameter groups:", {k: round(v ** 0.5 / max(n, 1), 5) for k, v in grads.items()}, flush=True)


def overfit(iso, mode, steps=300, N=96, bs=16):
    args, idx = setup("runs/v11/train/v9_run2_4090", iso)
    ds = M.V11Data(idx, "train", args, False, mode="true", limit=2000)
    b = next(iter(torch.utils.data.DataLoader(ds, N, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(0))))
    b2 = None
    if mode == "cmpcam":  # the same samples with the references of the moved-camera library
        a2 = argparse.Namespace(**vars(args))
        a2.refvis = "/root/pace/runs/v11/refvis" if "orig" in args.refvis else "/root/pace/runs/v11/refvis_orig"
        ds2 = M.V11Data(idx, "train", a2, False, mode="true", limit=2000)
        assert ds2.items == ds.items
        b2 = next(iter(torch.utils.data.DataLoader(ds2, N, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(0))))
    run = M.Runner(args)
    sim0 = run.pace.set_exp

    def set_exp(e):
        if e is not None:
            e = dict(e); e["scalars"] = e["scalars"].clone(); e["scalars"][..., 1] = 0  # the similarity scalar is not an image token
        return sim0(e)

    run.pace.set_exp = set_exp
    torch.manual_seed(0)
    Tgt = torch.randn(1, M.CHUNK, 7)
    Tgt = (0.25 * Tgt / Tgt.pow(2).mean().sqrt()).expand(N, -1, -1)
    names = sorted(set(b["lang"]))
    grp = torch.tensor([names.index(x) % 2 for x in b["lang"]])
    ip, im = (grp == 1).nonzero()[:, 0], (grp == 0).nonzero()[:, 0]
    print(f"[overfit] tasks {len(names)}, samples per side {len(ip)} {len(im)}", flush=True)
    pick = lambda x, ids: x[ids[torch.arange(N) % len(ids)]]
    vA, vB = pick(b["ref_vis"].float(), ip), pick(b["ref_vis"].float(), im)
    aA, aB = pick(b["a_ref"].float(), ip), pick(b["a_ref"].float(), im)
    if mode in ("blank", "cmpblank"):
        vA, vB = torch.zeros_like(vA), torch.zeros_like(vB)
    if mode == "cmpcam":
        assert not (b["ref_vis"] == b2["ref_vis"]).all()
        vA, vB = b["ref_vis"].float(), b2["ref_vis"].float()
    cmp, cap = None, {}
    params = run.params()
    if mode.startswith("cmp"):
        cmp = CompareHead().to(run.device)
        params = params + list(cmp.parameters())
        vt = run.vis_tokens

        def vis(*a, **k):
            cap["cur"] = vt(*a, **k)
            return cap["cur"]

        run.vis_tokens = vis
        run.head.register_forward_pre_hook(lambda m, inp: cap.__setitem__("h", inp[1].detach()))

    def forward(c):
        d = run(c)[0]
        return d if cmp is None else d + cmp(cap["cur"], c["ref_vis"].to(run.device).float(), cap["h"])

    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.0)
    W = M.weights(args, run.device)
    rng = np.random.default_rng(0)

    def batch(i, s):
        c = {k: (v[i] if torch.is_tensor(v) else [v[j] for j in i.tolist()]) for k, v in b.items()}
        if mode == "act":  # positive control: the sign is in the reference's action chunk, the images are held fixed
            c["ref_vis"] = vA[i]
            c["a_ref"] = torch.where(s[:, None, None] > 0, aA[i], aB[i])
        else:
            c["ref_vis"] = torch.where(s[:, None, None] > 0, vA[i], vB[i])
        c["target"] = s[:, None, None] * Tgt[i]
        return c

    def ev():
        run.adapter.eval()
        cs, rm = [], []
        with torch.no_grad():
            for i0 in range(0, 32, bs):
                i = torch.arange(i0, min(32, i0 + bs))
                for sign in (1.0, -1.0):
                    s = torch.full((len(i),), sign)
                    d = forward(batch(i, s))[:, :10].float().cpu()
                    t = (s[:, None, None] * Tgt[i])[:, :10]
                    cs.append(((d * t).flatten(1).sum(1) / (d.flatten(1).norm(dim=1) * t.flatten(1).norm(dim=1) + 1e-9)))
                    rm.append((d ** 2).mean((1, 2)).sqrt())
        run.adapter.train()
        return f"cos {float(torch.cat(cs).mean()):+.3f}, Delta rms {float(torch.cat(rm).mean()):.4f}"

    print(f"[overfit] isolation {iso}, {mode}, step 0: {ev()}", flush=True)
    for step in range(1, steps + 1):
        i = torch.as_tensor(rng.choice(N, bs, replace=False))
        s = torch.as_tensor(rng.choice([-1.0, 1.0], bs), dtype=torch.float32)
        c = batch(i, s)
        d = forward(c)
        loss = (((d - c["target"].to(run.device)) ** 2).mean(-1) * W).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"[overfit] isolation {iso}, {mode}, step {step}: loss {float(loss):.4f} (zero output {0.25 ** 2:.4f}), {ev()}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "look":
        look(sys.argv[2], sys.argv[3])
    else:
        overfit(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
