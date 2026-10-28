#!/bin/bash
# section 7.2 test (10-23, 4090): staging + pi0.5 / staging + experience on libero_10_swap. Log: logs/v10pro_stage.log
source /root/pace/env_pro_4090.sh
export PYTHONPATH=/root/pace/LIBERO-PRO:/root/pace/scripts/v10pro MUJOCO_EGL_DEVICE_ID=0
cd /root/pace/scripts/v10pro
python pro_stage.py --workers 12
echo STAGE_DONE
