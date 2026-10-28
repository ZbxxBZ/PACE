#!/bin/bash
# section 7.3 test (10-23, 4090): proprioceptive frame shift on libero_10_swap. Log: logs/v10pro_shift.log
source /root/pace/env_pro_4090.sh
export PYTHONPATH=/root/pace/LIBERO-PRO:/root/pace/scripts/v10pro MUJOCO_EGL_DEVICE_ID=0
cd /root/pace/scripts/v10pro
python pro_shift.py --workers 12
echo SHIFT_DONE
