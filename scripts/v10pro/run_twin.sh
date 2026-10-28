#!/bin/bash
# section 7.4 test (10-23, 4090): counterfactual twin observation on libero_10_swap. Log: logs/v10pro_twin.log
source /root/pace/env_pro_4090.sh
export PYTHONPATH=/root/pace/LIBERO-PRO:/root/pace/scripts/v10pro MUJOCO_EGL_DEVICE_ID=0
cd /root/pace/scripts/v10pro
python pro_twin.py --workers 12
echo TWIN_DONE
