#!/bin/bash
# V10 true-d upper bound, full run (10-23): 20 camera variants x init states 1..5, 3 sources each, 3 conditions;
# CPU (EGL rendering on GPU 0) -> runs/rectest/v10_shift/oracle_shift.json, log logs/v10_oracle_shift.log
source /root/pace/env_plus.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/PACE/code_v10
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
cd /root/pace
python scripts/v10/v10_oracle_shift.py --workers 14 > logs/v10_oracle_shift.log 2>&1
echo "[$(date '+%F %T')] ORACLE_SHIFT_DONE rc=$?" >> logs/v10_oracle_shift.log
