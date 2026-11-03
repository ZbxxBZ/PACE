# 11-03 (user): after the closed loop, continue V11b from pma_30k to 40000 steps on GPU1. The 30000-step run stays as it is: this one starts from a copy of
# its last.pt in runs/v11b/train/pma_40k. The cosine schedule is now over 40000 steps, so from step 30000 the learning rate rises from 0 to about 14% of the peak over 500 steps, then follows the cosine to 0 at 40000.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
Q=logs/queue_run.log
OUT=runs/v11b/train/pma_40k
now() { date '+%m-%d %H:%M:%S'; }
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
until grep -q "QUEUE END v11b_closed" $Q; do sleep 120; done
echo "$(now) QUEUE START v11b_40k" >> $Q
mkdir -p $OUT
[ -e $OUT/last.pt ] || cp runs/v11b/train/pma_30k/last.pt runs/v11b/train/pma_30k/best.pt runs/v11b/train/pma_30k/log.jsonl $OUT/
for try in 0 1 2 3 4 5; do
  while true; do
    used=$(gpu)
    [ $((used + 17000)) -le 44000 ] && [ ! -e logs/watchdog_g1.paused ] && break
    echo "$(now) v11b_40k waits: GPU1 ${used} MiB used" >> $Q; sleep 300
  done
  if [ "$used" -lt 1000 ]; then C="--bs 32 --accum 1 --workers 8"; big=1; else C="--bs 16 --accum 2 --workers 4"; big=0; fi
  echo "$(now) START v11b_40k (try $try) $C" >> $Q
  timeout 16h python scripts/v11/train_v11b.py --out $OUT --align pma --steps 40000 --eval_every 1000 --save_every 5000 --ramp_from 30000 --ramp 500 $C >> logs/v11b_40k.log 2>&1 &
  pid=$!
  base=0; t=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30; t=$((t + 30)); u=$(gpu)
    if [ $t -le 900 ]; then
      [ "$u" -gt "$base" ] && base=$u
    elif [ $big = 1 ] && [ "$u" -gt $((base + 3000)) ]; then
      echo "$(now) v11b_40k: someone else on GPU1 (${u} MiB, ours about ${base}), going back to the smaller batch" >> $Q
      kill -TERM $pid; wait $pid; break
    fi
  done
  wait $pid
  rc=$?
  echo "$(now) END v11b_40k rc=$rc (peak about ${base} MiB)" >> $Q
  [ $rc = 0 ] && break
  sleep 120
done
echo "$(now) QUEUE END v11b_40k rc=$rc" >> $Q
