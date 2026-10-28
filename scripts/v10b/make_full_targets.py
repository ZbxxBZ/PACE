"""
All LIBERO-plus libero_10 variants as closed-loop targets (10-25): task id, base task stem, the variant's own
instruction, category -- the format of select_noncam.py, for closed_loop_full.py (the main-score protocol: every
variant once from init state 0).
    python make_full_targets.py -> runs/rectest/v10_shift/full_targets.json
"""
import collections
import json

import collect_pi05 as C

OUT = "/root/pace/runs/rectest/v10_shift/full_targets.json"


def main():
    from lerobot.envs.libero import _get_suite
    from pace.eval_suite import classification

    base = _get_suite("libero_10")
    cls = classification("libero_10")
    st = C.stems()
    out = []
    for tid in range(base.n_tasks):
        t = base.get_task(tid)
        b = max((s for s in st if t.name.startswith(s)), key=len, default=None)
        out.append(dict(task_id=tid, name=t.name, base=b, language=t.language,
                        category=cls.get(t.name, (None,))[0]))
    json.dump(out, open(OUT, "w"), indent=1)
    print(len(out), "variants:", dict(collections.Counter(x["category"] for x in out)), "no base:",
          sum(x["base"] is None for x in out), "->", OUT)


if __name__ == "__main__":
    main()
