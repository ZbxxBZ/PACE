# 11-01 evening: after v11b_q3, the compare head's overfit test for 1500 steps (300 were still rising, below the 0.9 asked for). About 14 GB on GPU1.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
Q=logs/queue_run.log
L=logs/v11b_img.log
until grep -q "QUEUE END v11b_q3" $Q; do sleep 60; done
for a in "overfit off cmp 1500" "overfit off cmpcam 1500"; do
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ $((used + 15000)) -le 44000 ] && [ ! -e logs/watchdog_g1.paused ] && break
    sleep 120
  done
  echo "== v4 $a" >> $L
  timeout 60m python scripts/v11/v11_imgpath.py $a >> $L 2>&1 || echo "FAILED $a rc=$?" >> $L
done
echo "== done6" >> $L
