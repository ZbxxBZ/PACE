#!/bin/bash
# 10-22 (user's decision): once the V9 v2 closed-loop eval (scripts/v9b/eval_v9b.sh, GPU 0 and 1) has ended and GPU 0
# is free (< 1.5 GB in use), run V10-0 on GPU 0 of this 2x3090: smoke (must reproduce A's round 1), then the formal run
# (scripts/v10/eval_v10f.sh -> runs/eval/smoke_plus_V10f, runs/eval/plus_l10_V10f_acc3; log logs/eval_v10f_driver.log).
L=/root/pace/logs/eval_v10f_driver.log
while pgrep -f "[e]val_v9b.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] V9b eval ended: $(tail -n 1 /root/pace/logs/eval_v9b_driver.log)" >> $L
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -ge 1500 ]; do
  echo "[$(date '+%F %T')] GPU 0 busy, waiting" >> $L; sleep 300; done
GPU=0 exec bash /root/pace/scripts/v10/eval_v10f.sh
