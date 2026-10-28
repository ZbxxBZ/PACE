#!/bin/bash
# pi0.5 on LIBERO-PRO (10-23, 4090): the original libero_10 (a sanity check of the bridge, and the experience = the
# actions of its successes) and the position-swap libero_10_swap, 10 tasks x init states 0..9 each.
# Needs the server (start_server.sh). Log: logs/v10pro_base.log
source /root/pace/env_pro_4090.sh
export PYTHONPATH=/root/pace/LIBERO-PRO:/root/pace/scripts/v10pro MUJOCO_EGL_DEVICE_ID=0
cd /root/pace
python scripts/v10pro/pro_eval.py --suite libero_10 --save_actions --out runs/rectest/v10_pro/exp_l10 --workers 10
python scripts/v10pro/pro_eval.py --suite libero_10_swap --out runs/rectest/v10_pro/base_l10_swap --workers 10
echo RUN_BASE_DONE
