# 11-01 (user: "能不能再做完这两个任务继续在队列里加v11b的必要前置"; GPU1, container pace_g1; never push out other users' jobs).
# Starts when v11b_q1 has ended. V11b's data, needed whatever q1 says: pi0.5 rollouts on non-camera variants outside the test; replays
# with the camera in place (keep, pure); replays with wide camera changes. No training, no closed loop. One replay pool at a time
# (8 workers ~40 GB RAM). Ends with a data check.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b
Q=logs/queue_run.log
O=runs/v11b
now() { date "+%m-%d %H:%M:%S"; }
room() {  # GPU GB this stage needs: wait while GPU1 used + need > 44 GB, host MemAvailable < 60 GB or the watchdog has paused a job
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    host=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    [ $((used + $1 * 1000)) -le 44000 ] && [ "$host" -ge 60 ] && [ ! -e logs/watchdog_g1.paused ] && break
    echo "$(now) $2 waits: GPU1 ${used} MiB used, host available ${host} GB" >> $Q; sleep 300
  done
}
stage() {  # name need_gb limit_min stall_s command...
  local name=$1 need=$2 lim=$3 stall=$4; shift 4
  local log=logs/$name.log
  room $need $name; touch $log
  echo "$(now) START $name" >> $Q
  setsid timeout ${lim}m "$@" >> $log 2>&1 &
  local pid=$! age
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    [ -e logs/watchdog_g1.paused ] && touch $log  # a paused job is not a stalled one
    age=$(( $(date +%s) - $(stat -c %Y $log) ))
    if [ $age -gt $stall ]; then echo "$(now) STALL $name (${age}s)" >> $Q; kill -- -$pid; sleep 15; kill -9 -- -$pid; fi
  done
  wait $pid; local rc=$?
  echo "$(now) END $name rc=$rc" >> $Q
  return $rc
}
again() {  # stage, run again (they are resumable) if the memory watchdog stopped it: at most 3 more times, 10 min apart
  local n rc before
  for n in 0 1 2 3; do
    before=$(wc -l < logs/watchdog_g1.stopped 2>/dev/null || echo 0)
    stage "$@" && return 0
    rc=$?
    [ "$(wc -l < logs/watchdog_g1.stopped 2>/dev/null || echo 0)" -gt "$before" ] || return $rc
    echo "$(now) $1 stopped by the watchdog, again in 10 min" >> $Q; sleep 600
  done
  return $rc
}
until grep -q "QUEUE END v11b_q1" $Q && grep -q "END v11b_out2 rc=" $Q; do sleep 60; done
echo "$(now) QUEUE START v11b_q2" >> $Q
lane_noncam() {
  again v11b_noncam 10 400 2400 python scripts/v11/v11b_noncam.py --out $O
}
lane_replay() {
  again v11b_keep 2 300 2400 python scripts/v11/v11b_replay.py --mode keep --out $O || return 1
  again v11b_pure 10 300 2400 python scripts/v11/v11b_replay.py --mode pure --out $O || return 1
  again v11b_wide 10 500 2400 python scripts/v11/v11b_replay.py --mode wide --out $O
}
lane_noncam & a=$!
lane_replay & b=$!
wait $a; ra=$?; wait $b; rb=$?
stage v11b_data_check 1 60 1800 python scripts/v11/v11b_data_check.py $O; rc=$?
echo "$(now) QUEUE END v11b_q2 noncam=$ra replay=$rb check=$rc" >> $Q
