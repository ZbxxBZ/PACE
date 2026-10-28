#!/bin/bash
# V10-0 control (10-22, user's decision): the same protocol as the main V10-0 run (plus_l10_V10f_acc3) with
# --follow.cross_only 1: never follow the variant's own earlier successes (exact scene matches), only other variants'
# (the cross-perturbation transfer alone, without the deterministic replays of a variant's own success).
# Smoke first (round 1 must equal A), then the formal run -> runs/eval/plus_l10_V10f_cross_acc3; log
# logs/eval_v10f_cross_driver.log. GPU from $GPU (default 0); waits until it is free (< 1.5 GB).
# (A first start on GPU 1 at 16:17 was stopped during its smoke by the user: GPU 1 not to be used.)
G=${GPU:-0}
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -ge 1500 ]; do sleep 300; done
GPU=$G TAG=_cross EXTRA="--follow.cross_only 1" exec bash /root/pace/scripts/v10/eval_v10f.sh
