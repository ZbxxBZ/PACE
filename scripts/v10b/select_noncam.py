"""
Non-camera generalization test for V10-1b (10-24, registered before any result; user: does V10-1 help on LIBERO-plus's
other perturbation categories?). Same construction as the camera test: LIBERO-plus libero_10 variants of the same 8
base tasks (so the same reference sources apply: runs/v10b/noise_sources), init states 1..5 (the evaluation uses 0;
objects move 2-4 cm), env seed 1000 + 7 * task + init. Per category -- Light Conditions, Background Textures, Sensor
Noise, Language Instructions, Objects Layout, Robot Initial States -- 5 variants drawn at random (seed 2026) among
that category's variants of the 8 base tasks, excluding the variants the reference sources come from.
    python select_noncam.py -> runs/rectest/v10_shift/noncam_targets.json
"""
import collections
import json
import os

import numpy as np

R = "/root/pace/runs"
CATS = ("Light Conditions", "Background Textures", "Sensor Noise", "Language Instructions", "Objects Layout",
        "Robot Initial States")


def main():
    import libero.libero as L
    from lerobot.envs.libero import _get_suite
    from pace.eval_suite import classification

    suite = _get_suite("libero_10")
    cls = classification("libero_10")
    d = os.path.join(os.path.dirname(L.__file__), "init_files", "libero_10")
    stems = sorted(f[:-len(".pruned_init")] for f in os.listdir(d) if f.endswith(".pruned_init"))
    first = [json.loads(l) for l in open(f"{R}/rectest/v10_shift/base_targets/episodes.jsonl")]
    bases = {r["name"].split("_view_")[0] for r in first}
    src_ids = {json.loads(l)["task_id"] for l in open(f"{R}/v10b/noise_sources/episodes.jsonl")}
    cats = sorted({v[0] for v in cls.values() if v})
    print("categories in the classification:", cats)
    cand = collections.defaultdict(list)
    for tid in range(suite.n_tasks):
        t = suite.get_task(tid)
        cat = cls.get(t.name, (None,))[0]
        b = max((s for s in stems if t.name.startswith(s)), key=len, default=None)
        if cat in CATS and b in bases and tid not in src_ids:
            cand[cat].append(dict(task_id=tid, name=t.name, category=cat, base=b, language=t.language))
    rng = np.random.default_rng(2026)
    picked = []
    for cat in CATS:
        pool = sorted(cand[cat], key=lambda x: x["task_id"])
        idx = sorted(rng.choice(len(pool), min(5, len(pool)), replace=False))
        picked += [pool[i] for i in idx]
        print(f"{cat:24s} candidates {len(pool):4d} -> {[pool[i]['task_id'] for i in idx]}")
    json.dump(picked, open(f"{R}/rectest/v10_shift/noncam_targets.json", "w"), indent=1)
    print(f"-> {len(picked)} variants x init 1..5")


if __name__ == "__main__":
    main()
