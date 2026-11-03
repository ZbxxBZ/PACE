# 11-02: V11b's 3000-step short training, PMA then reading by time, one after the other on GPU1. About 16.7 GB GPU, 17 GB host each.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
Q=logs/queue_run.log
now() { date '+%m-%d %H:%M:%S'; }
echo "$(now) QUEUE START v11b_short" >> $Q
for arm in pma time; do
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ $((used + 17000)) -le 44000 ] && [ ! -e logs/watchdog_g1.paused ] && break
    echo "$(now) v11b_$arm waits: GPU1 ${used} MiB used" >> $Q; sleep 300
  done
  echo "$(now) START v11b_$arm" >> $Q
  timeout 8h python scripts/v11/train_v11b.py --out runs/v11b/train/$arm --align $arm > logs/v11b_$arm.log 2>&1
  rc=$?
  echo "$(now) END v11b_$arm rc=$rc" >> $Q
  [ $rc = 0 ] || break
done
echo "$(now) QUEUE END v11b_short" >> $Q
