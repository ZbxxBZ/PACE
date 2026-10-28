import collections, json
T = json.load(open("/root/pace/runs/rectest/v10_shift/noncam_targets.json"))
cat = {int(t["task_id"]): t["category"] for t in T}
name = {int(t["task_id"]): t.get("name", "") for t in T}
done = {json.loads(l)["key"] for l in open("/root/pace/runs/v10b/closed/noncam/episodes.jsonl")}
v = [l for l in open("/root/pace/runs/v10b/closed/noncam/episodes.jsonl") if '"v10v"' in l]
print("v10v done", len(v))
# expected v10v keys: each target x init 1..5 x its sources (count from the earlier v10 run of the same cells)
exp = collections.Counter()
for l in open("/root/pace/runs/v10b/closed/noncam/episodes.jsonl"):
    x = json.loads(l)
    if x["mode"] == "v10":
        k = "v10v" + x["key"][3:]
        if k not in done:
            exp[cat[x["task_id"]]] += 1
print("remaining v10v by category:", dict(exp), "total", sum(exp.values()))
sn = collections.Counter()
for l in open("/root/pace/runs/v10b/closed/noncam/episodes.jsonl"):
    x = json.loads(l)
    if x["mode"] == "v10" and "v10v" + x["key"][3:] not in done and cat[x["task_id"]] == "Sensor Noise":
        sn[name[x["task_id"]][-12:]] += 1
print("remaining sensor-noise variants:", dict(sn))
