#!/bin/bash
# V10-0 (non-learned follower, pace/follow.py) on LIBERO-plus libero_10 (10-22; the 2x3090, or the 3x3090 box
# the 3x3090 box), with the protocol of A, V5, V7, V9 on the 2x3090: the store starts empty and every finished batch's rollouts join it
# (--pace.accumulate; --pace.condition none with --pace.record_features: the base policy plus the follower, which
# reads the run's own stored successes), 3 rounds over the 140 variants of plus_l10_sample20.json, batch 5, one GPU.
# The base is A's run of 10-15 on the 2x3090 (plus_l10_A_r3, copied here); this box is the same GPU model, and the
# smoke checks that it reproduces A: round 1 of the smoke (one batch, empty store: nothing to follow) must equal A's
# smoke episode by episode.
#   GPU=0 SMOKE_ONLY=1 bash eval_v10f.sh     # smoke only
# TAG=_g1 suffixes the output dirs and logs (several runs at once); EXTRA passes more eval_suite options
# (e.g. EXTRA="--follow.cross_only 1"). This box's /dev/dri nodes are moved away (10-22: a dead GPU's DRM node
# hangs every drm_open on the host); NVIDIA EGL renders through /dev/nvidia* without them.
# 3x3090 box: env_plus_box3.sh = env_plus.sh + the ImageMagick copied under /root/pace/sys (the container has none)
BOX3=0; [ -f /root/pace/env_plus_box3.sh ] && BOX3=1
if [ $BOX3 = 1 ]; then source /root/pace/env_plus_box3.sh; else source /root/pace/env_plus.sh; fi
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
CODE=/root/pace/PACE/code_v10
export PYTHONPATH=/root/pace/LIBERO-plus:$CODE
cd $CODE
G=${GPU:-0}
T=${TAG:-}
LOG=/root/pace/logs/eval_v10f${T}_driver.log
[ $BOX3 = 0 ] || [ -z "$(ls -A /dev/dri 2>/dev/null)" ] ||
  { echo "/dev/dri is not empty: EGL would hang in drm_open" >> $LOG; exit 1; }
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
fail() { say "$1"; exit 1; }
keep_old() { [ -e "$1" ] && mv "$1" "$1.old_$(date +%m%d%H%M)"; return 0; }
E=/root/pace/runs/eval
COMMON="--env_type libero_plus --suite libero_10 --batch_size 5"
PACE="--pace.condition none --pace.accumulate --pace.record_features --pace.sources rollout --follow 1 $EXTRA"
run() {  # out_dir log args...
  local out=$1 log=$2; shift 2
  CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G python -m pace.eval_suite $COMMON --out_dir $out "$@" > $log 2>&1
}
say "start on GPU $G (code $CODE) tag [$T] extra [$EXTRA]"

# --- 1. smoke ---
S=$E/smoke_plus_V10f$T; keep_old $S
run $S /root/pace/logs/eval_smoke_plus_V10f$T.log --tasks_file $E/_old/plus_l10_sample5_smoke.json --rounds 2 $PACE ||
  fail "SMOKE_V10F_FAILED (eval_smoke_plus_V10f$T.log)"
python - $E/_old/smoke_plus_A $S >> $LOG 2>&1 <<'PY' || fail "SMOKE_CHECK_FAILED"
import json, sys
load = lambda d: {(r["round"], r["task_id"]): r for r in map(json.loads, open(d + "/episodes.jsonl"))}
a, v = load(sys.argv[1]), load(sys.argv[2])
assert len(v) == 10, len(v)
r1 = [k for k in a if k[0] == 1]
diff = [(k, a[k]["success"], v[k]["success"], a[k]["env_steps"], v[k]["env_steps"]) for k in r1
        if (a[k]["success"], a[k]["env_steps"]) != (v[k]["success"], v[k]["env_steps"])]
assert not diff, "round 1 of V10-0 (nothing to follow) differs from A's smoke: " + str(diff)
assert all(v[k]["follow"]["status"] == "base" for k in r1), [v[k]["follow"] for k in r1]
print("SMOKE CHECK OK: round 1 identical to A's smoke (this box reproduces A)")
for k in sorted(k for k in v if k[0] == 2):
    f = v[k]["follow"]
    src = v.get((1, f.get("ref_task_id"))) if f.get("ref_task_id") is not None else None
    print(f"  round 2 task {k[1]}: success {v[k]['success']} steps {v[k]['env_steps']} follow {f.get('status')} "
          f"kind {f.get('ref_kind')} from task {f.get('ref_task_id')} (its round-1 steps "
          f"{src['env_steps'] if src else None}) followed {f.get('followed_calls')} calls; A round 2: "
          f"success {a[k]['success']} steps {a[k]['env_steps']}")
PY
say "SMOKE_OK"
[ "$SMOKE_ONLY" = 1 ] && { say "SMOKE_ONLY: stopping before the formal run"; exit 0; }

# --- 2. formal ---
O=$E/plus_l10_V10f${T}_acc3; keep_old $O
say "formal -> $O (GPU $G)"
run $O /root/pace/logs/eval_plus_l10_V10f$T.log --tasks_file $E/plus_l10_sample20.json --rounds 3 $PACE
r=$?
say "FORMAL_DONE rc=$r"
python -m pace.compare_runs $E/plus_l10_A_r3 $O --by category --out $E/compare_A_V10f$T.json > /root/pace/logs/compare_A_V10f$T.log 2>&1
say "COMPARE done (logs/compare_A_V10f$T.log)"
say "EVAL_V10F_DONE"
