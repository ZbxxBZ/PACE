#!/bin/bash
# V10 premise 2: open-loop replay of same-layout successes into camera-viewpoint variants (GPU 0 for rendering only).
source /root/pace/env_plus.sh
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/PACE/code_v9b
cd /root/pace/PACE/code_v9b
python /root/pace/scripts/v10/v10_oracle_replay.py > /root/pace/logs/v10_oracle_replay.log 2>&1
echo "ORACLE_DONE rc=$?" >> /root/pace/logs/v10_oracle_replay.log
