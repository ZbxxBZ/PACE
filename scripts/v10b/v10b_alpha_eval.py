"""
Offline check of run1's follow decision on the validation scenes (10-24). For every validation pair episode (expert
and negative; perturbed ones carry no follow label), the model's two heads at the first call (c = 0, the decision
"follow this reference or not" at the start) and averaged over the first --early calls. Decision: follow when
P(follow succeeds) - P(pi0.5 succeeds) > m. Outcome if followed = the pair's follow label (true-d following: optimistic,
d is assumed right), else = pi0.5's own outcome under the same augmentation. Reported per m: success rate, pi0.5 alone,
damage (pi0.5 would succeed but the followed pair fails) among pi0.5 successes, rescue among pi0.5 failures, how often
it follows; plus the ideal decision.
    python v10b_alpha_eval.py runs/v10b/train/run1/best.pt
"""
import collections
import json
import sys

import numpy as np
import torch

import train_v10b as T


def main():
    ck = torch.load(sys.argv[1], weights_only=False)
    a = ck["args"]
    args = type("A", (), dict(a, out=a["out"]))()
    early = 3
    idx = T.build_index(args)
    from pace.pi05 import InstructionText

    text = InstructionText.from_checkpoint("lerobot/pi05_libero_finetuned", tasks=list(idx["lang"].values()))
    lang_vec = {t: text.embed(l).float() for t, l in idx["lang"].items()}
    ds = T.PairData(idx, "val", args, lang_vec, False)
    keep = [i for i, it in enumerate(ds.items) if idx["eps"][idx["samples"][it]["ep"]]["kind"] != "perturbed"
            and idx["samples"][it]["c"] < early]
    sub = torch.utils.data.Subset(ds, keep)
    dl = torch.utils.data.DataLoader(sub, 64, num_workers=8)
    D = np.load(f"{args.feats}/pca.npz")["comp"].shape[0]
    model = T.Comparator(D, w=args.width, pool=args.pool).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    pf, pb = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, lf, lb = model(b["feats"], b["state"], b["lang"], b["drop"], args.d_scale)
            pf.append(torch.sigmoid(lf.float()).cpu())
            pb.append(torch.sigmoid(lb.float()).cpu())
    pf, pb = torch.cat(pf).numpy(), torch.cat(pb).numpy()
    ep_scores = collections.defaultdict(list)
    for k, i in enumerate(keep):
        s = idx["samples"][ds.items[i]]
        ep_scores[s["ep"]].append((s["c"], pf[k], pb[k]))
    rows = []
    for e, sc in ep_scores.items():
        ep = idx["eps"][e]
        if ep["base"] is None:
            continue
        sc.sort()
        rows.append(dict(kind=ep["kind"], follow=bool(ep["follow"]), base=bool(ep["base"]),
                         m0=sc[0][1] - sc[0][2], me=float(np.mean([x[1] - x[2] for x in sc]))))
    print(f"[val episodes] {len(rows)}: " + ", ".join(f"{k} {sum(r['kind'] == k for r in rows)}" for k in ("expert", "negative")))
    base = np.array([r["base"] for r in rows])
    fol = np.array([r["follow"] for r in rows])
    print(f"pi0.5 alone {100 * base.mean():.1f}% | ideal (follow only when pi0.5 fails and following succeeds) "
          f"{100 * (base | fol).mean():.1f}% | always follow {100 * fol.mean():.1f}%")
    for key in ("m0", "me"):
        print(f"-- decision score {key} ({'first call' if key == 'm0' else f'mean of first {early} calls'})")
        s = np.array([r[key] for r in rows])
        for m in (-0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
            go = s > m
            out = np.where(go, fol, base)
            dmg = (base & go & ~fol).sum() / max(base.sum(), 1)
            res = (~base & go & fol).sum() / max((~base).sum(), 1)
            neg = np.array([r["kind"] == "negative" for r in rows])
            print(f"  m {m:+.2f}: success {100 * out.mean():5.1f}%  follows {100 * go.mean():5.1f}%  "
                  f"damage {100 * dmg:4.1f}%  rescue {100 * res:4.1f}%  negatives followed {100 * go[neg].mean():4.1f}%")


if __name__ == "__main__":
    main()
