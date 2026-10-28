"""Print each camera target's BDDL goal predicates and the simulator body names (for the corrected oracle's anchors)."""
import json
import sys

sys.argv = sys.argv[:1]
import v10_oracle_shift as O  # noqa: E402

O.worker_init()
seen = {}
for tid in (684, 732, 787, 840, 895, 934, 968, 1080):
    env = O.make_env(tid, 0)
    try:
        env.reset(seed=0)
        inner = env._env.env
        goal = inner.parsed_problem["goal_state"]
        print(tid, json.dumps(goal), "| bodies:", sorted(inner.obj_body_id), "| movable:", sorted(inner.objects_dict))
    finally:
        env.close()
