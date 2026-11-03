"""11-01 confirmation of the out-of-frame split: every camera variant with a goal object's centre outside agentview at reset,
not among the 20 test or the 40 tilt-check variants, with a same-task reference. No selection by outcome."""
import json, os, re, collections
import numpy as np
P = "/root/pace/runs"
A = {int(k): v for k, v in json.load(open(f"{P}/v11/closed/lr5/inview_all.json")).items()}
F = {x["task_id"]: x for x in json.load(open(f"{P}/rectest/v10_shift/full_targets.json"))}
test = {json.loads(l)["task_id"] for l in open(f"{P}/rectest/v10_shift/base_targets/episodes.jsonl")}
tilt = {x["task_id"] for x in json.load(open(f"{P}/v11/closed/tilt/targets.json"))}
own = collections.Counter(r["base"] for r in map(json.loads, open(f"{P}/v10b/noise_sources/episodes.jsonl")) if r["success"])
fb = {}
for l in open(f"{P}/v10b/closed/full_base/episodes.jsonl"):
    r = json.loads(l); fb[r["task_id"]] = float(r["success"])
clean = lambda s: re.sub(r" view \d+ \d+ \d+ \d+ \d+ initstate \d+$", "", s)
def group(n):
    h, e, sc, er, ev = [int(x) for x in re.search(r"view_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_initstate", n).groups()]
    return "distance" if sc != 100 else ("tilt" if h == 0 and e == 0 else "orbit")
T, noref = [], 0
for t, v in sorted(A.items()):
    if v["category"] != "Camera Viewpoints" or t in test or t in tilt:
        continue
    o = [n for n in v["goal"] if n in v["px"] and not (0 <= v["px"][n][0] < 360 and 0 <= v["px"][n][1] < 360)]
    if not o:
        continue
    if own[v["base"]] == 0:
        noref += 1
        continue
    T.append(dict(F[t], language=clean(F[t]["language"]), out=o, group=group(v["name"])))
assert not any(" view " in x["language"] for x in T)
os.makedirs(f"{P}/v11/closed/outframe", exist_ok=True)
json.dump(T, open(f"{P}/v11/closed/outframe/targets.json", "w"), indent=0)
print("variants", len(T), "without own reference", noref, "groups", collections.Counter(x["group"] for x in T), "objects", collections.Counter(n for x in T for n in x["out"]))
print("full-run base", round(float(np.mean([fb[x["task_id"]] for x in T])) * 100, 1), "by group", {g: round(float(np.mean([fb[x["task_id"]] for x in T if x["group"] == g])) * 100, 1) for g in {x["group"] for x in T}})
