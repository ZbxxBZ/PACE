# 11-02: V11b full training with PMA on GPU1 only, 30000 steps (user, 11-02), a checkpoint every 5000. Card ours alone: --bs 32 --accum 1 --workers 8 (11-02, user: more memory is fine
# when only we are on the card). If someone else comes to the card the guard stops it; it goes on from the last validation at --bs 16 --accum 2 (about 16.7 GB).
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
Q=logs/queue_run.log
now() { date '+%m-%d %H:%M:%S'; }
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
echo "$(now) QUEUE START v11b_full" >> $Q
for try in 0 1 2 3 4 5; do
  while true; do
    used=$(gpu)
    [ $((used + 17000)) -le 44000 ] && [ ! -e logs/watchdog_g1.paused ] && break
    echo "$(now) v11b_full waits: GPU1 ${used} MiB used" >> $Q; sleep 300
  done
  if [ "$used" -lt 1000 ]; then C="--bs 32 --accum 1 --workers 8"; big=1; else C="--bs 16 --accum 2 --workers 4"; big=0; fi
  echo "$(now) START v11b_full (try $try) $C" >> $Q
  timeout 40h python scripts/v11/train_v11b.py --out runs/v11b/train/pma_30k --align pma --steps 30000 --eval_every 1000 --save_every 5000 $C >> logs/v11b_30k.log 2>&1 &
  pid=$!
  base=0; t=0
  while kill -0 $pid 2>/dev/null; do
    sleep 30; t=$((t + 30)); u=$(gpu)
    if [ $t -le 900 ]; then
      [ "$u" -gt "$base" ] && base=$u
    elif [ $big = 1 ] && [ "$u" -gt $((base + 3000)) ]; then
      echo "$(now) v11b_full: someone else on GPU1 (${u} MiB, ours about ${base}), going back to the smaller batch" >> $Q
      kill -TERM $pid; wait $pid; break
    fi
  done
  wait $pid
  rc=$?
  echo "$(now) END v11b_full rc=$rc (peak about ${base} MiB)" >> $Q
  [ $rc = 0 ] && break
  sleep 120
done
if [ $rc = 0 ]; then
  for c in last.pt best.pt; do
    python scripts/v11/v11b_swap.py runs/v11b/train/pma_30k $c > logs/v11b_30k_swap_$c.log 2>&1
    cp runs/v11b/train/pma_30k/swap.json runs/v11b/train/pma_30k/swap_$c.json
  done
fi
echo "$(now) QUEUE END v11b_full rc=$rc" >> $Q
