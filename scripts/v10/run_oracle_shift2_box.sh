#!/bin/bash
# V10 true-d upper bound with corrected anchors (10-23) on the 3x3090 box: CPU work, EGL rendering on GPU 1 only.
# Waits until the V10-0 run (the sources) has fully arrived from the 2x3090 (87 files, 3006175473 bytes), then
# runs scripts/v10/v10_oracle_shift2.py -> runs/rectest/v10_shift/oracle_shift2.json, log logs/v10_oracle_shift2.log.
P=/root/pace; LOG=$P/logs/v10_oracle_shift2.log; R=$P/runs/eval/plus_l10_V10f_acc3
cd $P
[ -z "$(ls -A /dev/dri 2>/dev/null)" ] || { echo "/dev/dri is not empty: EGL would hang in drm_open" > $LOG; exit 1; }
until [ "$(find $R -type f 2>/dev/null | wc -l)" = 87 ] && [ "$(du -sb $R 2>/dev/null | cut -f1)" = 3006175473 ]; do
  sleep 30
done
echo "[$(date '+%F %T')] sources complete" > $LOG
source $P/env_plus_box3.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=$P/LIBERO-plus:$P/PACE/code_v10
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
mkdir -p $P/runs/rectest/v10_shift
python scripts/v10/v10_oracle_shift2.py --workers ${WORKERS:-24} >> $LOG 2>&1
rc=$?; echo "[$(date '+%F %T')] ORACLE_SHIFT2_DONE rc=$rc" >> $LOG
