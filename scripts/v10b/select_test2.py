"""
Second generalization test set (10-24, registered before any result): the first set's 20 camera-viewpoint variants
were seen during the V10-1b closed loop, so later changes (alpha) are judged on 20 NEW camera variants of the same 8
base tasks, the same number per base task as the first set (stratified), drawn at random with seed 2026 among the
LIBERO-plus libero_10 "Camera Viewpoints" variants not in the first set; init states 1..5 as before (references and
seeds: same rule as the first set).
    python select_test2.py -> runs/rectest/v10_shift/test2_targets.json
"""
import collections
import json

import numpy as np

R = "/root/pace/runs"


def main():
    from lerobot.envs.libero import _get_suite
    from pace.eval_suite import classification

    suite = _get_suite("libero_10")
    cls = classification("libero_10")
    first = [json.loads(l) for l in open(f"{R}/rectest/v10_shift/base_targets/episodes.jsonl")]
    used = {r["task_id"] for r in first}
    base_of = {r["task_id"]: r["name"].split("_view_")[0] for r in first}
    per_base = collections.Counter(base_of[t] for t in used)
    lang = {r["name"].split("_view_")[0]: r["base"] for r in first}
    cand = collections.defaultdict(list)
    for tid in range(suite.n_tasks):
        name = suite.get_task(tid).name
        if cls.get(name, (None,))[0] != "Camera Viewpoints" or tid in used or "_view_" not in name:
            continue
        b = name.split("_view_")[0]
        if b in per_base:
            cand[b].append((tid, name))
    rng = np.random.default_rng(2026)
    picked = []
    for b in sorted(per_base):
        pool = sorted(cand[b])
        idx = rng.choice(len(pool), per_base[b], replace=False)
        for i in sorted(idx):
            picked.append(dict(task_id=int(pool[i][0]), name=pool[i][1], base=lang[b]))
        print(f"{b[:60]:60s} need {per_base[b]}, candidates {len(pool)}")
    json.dump(picked, open(f"{R}/rectest/v10_shift/test2_targets.json", "w"), indent=1)
    print(f"-> {len(picked)} variants: {[p['task_id'] for p in picked]}")


if __name__ == "__main__":
    main()
