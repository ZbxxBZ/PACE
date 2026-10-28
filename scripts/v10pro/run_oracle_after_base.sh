#!/bin/bash
# Gate 1 (10-23, 4090): wait until run_base.sh has finished (experience + base on libero_10_swap), then the true-d
# upper bound on libero_10_swap. Log: logs/v10pro_oracle.log
cd /root/pace
until grep -q RUN_BASE_DONE logs/v10pro_base.log 2>/dev/null; do sleep 30; done
source /root/pace/env_pro_4090.sh
export PYTHONPATH=/root/pace/LIBERO-PRO:/root/pace/scripts/v10pro MUJOCO_EGL_DEVICE_ID=0
python scripts/v10pro/pro_oracle.py --workers 16
echo ORACLE_PRO_DONE
