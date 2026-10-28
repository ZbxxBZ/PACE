import collections
import json
import os

import numpy as np

R = "/root/pace/runs/v10b/closed/explore"
r = [json.loads(l) for l in open(f"{R}/episodes.jsonl")]
print(f"episodes {len(r)}, success {sum(x['success'] for x in r)} ({100 * np.mean([x['success'] for x in r]):.1f}%), "
      f"val {sum(x['split'] == 'val' for x in r)}")
cnt = collections.Counter()
for x in r:
    f = f"{R}/decisions/{x['key']}.npz"
    if not os.path.exists(f):
        cnt["no_file"] += 1
        continue
    z = np.load(f)
    for k, g in zip(z["kind"], z["go"]):
        cnt[(str(k), bool(g), x["success"])] += 1
for kind in ("start", "boundary", "event", "empty", "stuck"):
    row = []
    for g in (True, False):
        s, f_ = cnt[(kind, g, True)], cnt[(kind, g, False)]
        row.append(f"{'take' if g else 'cont'} {s}/{s + f_}")
    print(f"{kind:8s}: " + "  ".join(row) + "  (successes / decisions)")
print("episodes without a decision file:", cnt["no_file"])
