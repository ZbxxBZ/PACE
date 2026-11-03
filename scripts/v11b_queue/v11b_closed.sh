# 11-02: stage 3 of V11b, after v11b_full.sh (30000-step training, then the swap probes): closed loop of best.pt on GPU1 only, camera then non-camera.
# Card ours alone: 16 envs per batch (about twice as fast, peak logged here). Someone else on the card: the guard stops it and it goes on at 8; episodes are resumable.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
Q=logs/queue_run.log
CK=runs/v11b/train/pma_30k/best.pt
now() { date '+%m-%d %H:%M:%S'; }
gpu() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
until grep -q "QUEUE END v11b_full rc=" $Q; do sleep 120; done
grep -q "QUEUE END v11b_full rc=0" $Q || { echo "$(now) v11b_closed not started: training failed" >> $Q; exit 1; }
stage() {  # name out extra args
  name=$1; out=$2; shift 2
  for try in 0 1 2 3 4 5; do
    while true; do
      used=$(gpu)
      [ $((used + 16000)) -le 44000 ] && [ ! -e logs/watchdog_g1.paused ] && break
      echo "$(now) $name waits: GPU1 ${used} MiB used" >> $Q; sleep 300
    done
    if [ "$used" -lt 1000 ] && [ "$try" = 0 ]; then B=16; else B=8; fi  # any failed try goes on at 8
    echo "$(now) START $name (try $try) batch $B" >> $Q
    timeout 16h python scripts/v11/closed_loop_v11b.py --ckpt $CK --out $out --refs_dir runs/v11b/closed/refs --batch $B "$@" >> logs/$name.log 2>&1 &
    pid=$!
    base=0; t=0
    while kill -0 $pid 2>/dev/null; do
      sleep 30; t=$((t + 30)); u=$(gpu)
      if [ $t -le 900 ]; then
        [ "$u" -gt "$base" ] && base=$u
      elif [ "$u" -gt "$base" ] && [ $B = 8 ]; then
        base=$u
      elif [ $B = 16 ] && [ "$u" -gt $((base + 3000)) ]; then
        echo "$(now) $name: someone else on GPU1 (${u} MiB, ours about ${base}), going back to 8" >> $Q
        kill -TERM $pid; wait $pid; break
      fi
    done
    wait $pid
    rc=$?
    echo "$(now) END $name rc=$rc (peak about ${base} MiB)" >> $Q
    [ $rc = 0 ] && return 0
    sleep 120
  done
  return 1
}
echo "$(now) QUEUE START v11b_closed" >> $Q
stage v11b_cl_camera runs/v11b/closed/run1 --modes v11 v11other base
a=$?
stage v11b_cl_noncam runs/v11b/closed/noncam --modes v11 base --targets runs/rectest/v10_shift/noncam_targets.json
b=$?
echo "$(now) QUEUE END v11b_closed camera=$a noncam=$b" >> $Q
