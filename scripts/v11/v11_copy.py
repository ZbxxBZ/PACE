"""How much of Delta is a pull toward the reference's chunk? And how much of the label? Executed 10 steps, validation, 'true' mode."""
import argparse, json
import numpy as np
import torch
import train_v11_v9 as M
T = M.T
args = argparse.Namespace(**json.load(open("runs/v11/train/v9_run2_4090/args.json")))
kinds = list(args.kinds)
args.kinds, args.dlabel, args.state = [k for k in kinds if k != "pi05_aug"], "true", "abs"
idx = T.build_index(args); args.kinds = kinds; idx = M.add_pi05_aug(idx)
ds = M.V11Data(idx, "val", args, False, mode="true", limit=400)
run = M.Runner(args)
ck = torch.load("runs/v11/train/v9_run2_4090/best.pt", map_location="cpu", weights_only=False)
run.adapter.load_state_dict(ck["adapter"]); run.head.load_state_dict(ck["head"]); run.adapter.eval()
Ds, Ts, Cs, Ks = [], [], [], []
with torch.no_grad():
    for b in torch.utils.data.DataLoader(ds, 8, shuffle=False, num_workers=2):
        Ds.append(run(b)[0][:, :10].float().cpu()); Ts.append(b["target"][:, :10].float()); Cs.append((b["a_ref"] - b["a_pi"])[:, :10].float()); Ks.append(b["kind"])
D, Y, C, K = [torch.cat(x).numpy() for x in (Ds, Ts, Cs, Ks)]
cos = lambda a, b: float(((a * b).reshape(len(a), -1).sum(1) / (np.linalg.norm(a.reshape(len(a), -1), axis=1) * np.linalg.norm(b.reshape(len(b), -1), axis=1) + 1e-9)).mean())
def slope(y, x):
    return float((y * x).sum() / (x * x).sum())
def r2(y, p):
    return 1 - float(((y - p) ** 2).sum() / (y ** 2).sum())
for nm, s in (("positive", K != M.KINDS.index("negative")), ("expert", K == 0), ("perturbed", K == 1), ("carry", K == 3), ("pi05_aug", K == 4), ("negative", K == 2)):
    d, y, c = D[s], Y[s], C[s]
    k = slope(d, c); ky = slope(y, c)
    print(f"{nm:9s} n={s.sum():3d} rms Delta {np.sqrt((d**2).mean()):.3f} label {np.sqrt((y**2).mean()):.3f} a_ref-a_pi {np.sqrt((c**2).mean()):.3f} | cos(Delta,label) {cos(d,y):+.3f} cos(Delta,copy) {cos(d,c):+.3f} cos(label,copy) {cos(y,c):+.3f} | slope Delta on copy {k:.3f}, label on copy {ky:.3f} | R2 of Delta by copy {r2(d, k*c):.3f}; R2 of label by Delta {r2(y,d):.3f}, by {ky:.2f}*copy {r2(y, ky*c):.3f}")
pos = K != M.KINDS.index("negative")
print("per axis (positive): slope of Delta on copy", [round(slope(D[pos][..., i], C[pos][..., i]), 3) for i in range(7)], "label", [round(slope(Y[pos][..., i], C[pos][..., i]), 3) for i in range(7)])
