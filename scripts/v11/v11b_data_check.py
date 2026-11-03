"""
Checks the V11b data (11-01): per mode episodes, samples, a_pi finite on call rows, frames and a_pi aligned; noncam variants outside
the test; wide cameras by kind; keep frames equal V11's clear replay in everything but the agentview; a contact sheet per mode.
    python v11b_data_check.py runs/v11b
"""
import collections, json, os, sys
import numpy as np
import imageio.v2 as imageio
import cv2

out = sys.argv[1]
V = "/root/pace/runs/v10b"
sheet, fail = [], 0
for m in ("keep", "pure", "wide", "noncam"):
    p = f"{out}/{m}.jsonl"
    if not os.path.exists(p):
        print(m, "missing"); fail += 1; continue
    rows = [json.loads(l) for l in open(p)]
    samples, bad, tiles = 0, 0, []
    for r in rows:
        k = r.get("fkey", r["key"])
        z = np.load(f"{out}/frames/{m}/{k}.npz")
        a = np.load(f"{out}/api/{m}/{k}.npy")
        t = z["t"]
        n = len(t) - 1 if len(a) == len(t) - 1 and not z["success"] and int(t[-1]) == 520 else len(t)  # ran out of steps: pi0.5 is not called on the last row
        calls = np.array([i for i in range(n) if int(t[i]) % 10 == 0])
        samples += len(calls)
        if len(a) != n or not np.isfinite(a[calls].astype(np.float32)).all():
            bad += 1
        if len(tiles) < 6 and len(calls) > 8:
            f = z["agent"][calls[8]]
            tiles.append(cv2.resize(np.ascontiguousarray(f[::-1, ::-1]), (224, 224)))
    print(f"{m}: {len(rows)} episodes, {samples} samples, {bad} bad", flush=True)
    fail += bad
    if m == "keep" and rows:
        r = rows[0]
        a = np.load(f"{out}/frames/keep/{r['fkey']}.npz")
        b = np.load(f"{V}/frames/{r['kind']}/{r['key']}_a0.npz")
        print("  keep vs stored: wrist equal", bool(np.array_equal(a["wrist"], b["wrist"])), "state equal", bool(np.array_equal(a["eef_pos"], b["eef_pos"])), "agent differs", bool((a["agent"] != b["agent"]).any()))
    if m == "wide":
        c = collections.Counter("orbit" if r["cam_wide"]["h"] else ("distance" if r["cam_wide"]["sc"] != 1.0 else "tilt") for r in rows)
        print("  camera kinds", dict(c))
    if m == "noncam":
        test = {x["task_id"] for x in json.load(open("/root/pace/runs/rectest/v10_shift/noncam_targets.json"))}
        src = {json.loads(l)["task_id"] for l in open(f"{V}/noise_sources/episodes.jsonl")}
        print("  overlap with test / sources", len({r["task_id"] for r in rows} & test), len({r["task_id"] for r in rows} & src),
              "success by category", {c: round(np.mean([r["success"] for r in rows if r["category"] == c]) * 100, 1) for c in {r["category"] for r in rows}}, "split", collections.Counter(r["split"] for r in rows))
    if tiles:
        while len(tiles) < 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet.append(np.concatenate(tiles, 1))
if sheet:
    imageio.imwrite(f"{out}/check.png", np.concatenate(sheet, 0))
print("data check", "FAILED" if fail else "ok", flush=True)
sys.exit(1 if fail else 0)
