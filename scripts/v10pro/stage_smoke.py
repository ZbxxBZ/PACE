import json

import numpy as np

import pro_oracle as P
import pro_stage as S

rows = [json.loads(l) for l in open(P.EXP + "/episodes.jsonl")]
arr = np.load(P.EXP + "/actions.npz")
for task in (1, 4):
    r = next(r for r in sorted(rows, key=lambda r: r["init"]) if r["task"] == task and r["success"])
    key = "t%d_i%d" % (task, r["init"])
    src = dict(key=key, task=task, init=r["init"], seed=r["seed"], actions=arr[key])
    src["rec"] = P.record_source(src)
    print("RES plan", task, [(p["kind"], p["t"], p["pre"], p["anchor"]) for p in S.plan(src["rec"])], flush=True)
    for m in ("exp", "vla"):
        print("RES", S.run_cell(((task, 0), src, m, 18765)), flush=True)
