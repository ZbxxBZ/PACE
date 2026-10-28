#!/bin/bash
# V10 true-d upper bound (10-22), the "base" condition: the base policy alone on the evaluation's 20 camera-viewpoint
# variants from init states 1..5 (the evaluation uses 0), one episode each, batch 5, GPU 0
# -> runs/rectest/v10_shift/base_targets (log logs/v10_shift_base.log). No PACE.
source /root/pace/env_plus.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
CODE=/root/pace/PACE/code_v10
export PYTHONPATH=/root/pace/LIBERO-plus:$CODE
cd $CODE
D=/root/pace/runs/rectest/v10_shift; mkdir -p $D
python3 - <<'PY'
import json
rows = [json.loads(l) for l in open("/root/pace/runs/eval/plus_l10_A_r3/episodes.jsonl")]
cam = sorted({r["task_id"] for r in rows if r["category"] == "Camera Viewpoints"})
json.dump(cam, open("/root/pace/runs/rectest/v10_shift/camera20.json", "w"))
print(len(cam), "camera variants")
PY
CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python -m pace.eval_suite --env_type libero_plus --suite libero_10 \
  --batch_size 5 --tasks_file $D/camera20.json --init_indices 1 2 3 4 5 --rounds 1 --out_dir $D/base_targets \
  > /root/pace/logs/v10_shift_base.log 2>&1
echo "[$(date '+%F %T')] SHIFT_BASE_DONE rc=$?" >> /root/pace/logs/v10_shift_base.log
