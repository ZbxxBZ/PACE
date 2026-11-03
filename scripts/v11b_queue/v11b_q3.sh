# 11-01 evening (user: "可以"): one job at a time on GPU1, about 14 GB; others on the card are never pushed out. First the comparison
# prototype on the overfit test, then the non-camera rollouts again (recording bug fixed), then the data check.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b
Q=logs/queue_run.log
L=logs/v11b_img.log
O=runs/v11b
now() { date "+%m-%d %H:%M:%S"; }
room() {  # wait while GPU1 used + 15 GB > 44 GB, host MemAvailable < 60 GB or the watchdog has paused a job
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    host=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
    [ $((used + 15000)) -le 44000 ] && [ "$host" -ge 60 ] && [ ! -e logs/watchdog_g1.paused ] && break
    echo "$(now) v11b_q3 waits: GPU1 ${used} MiB used, host available ${host} GB" >> $Q; sleep 120
  done
}
until grep -q "^== done4" $L; do sleep 30; done
echo "$(now) QUEUE START v11b_q3" >> $Q
for a in "overfit off cmp" "overfit off cmpblank" "overfit off cmpcam"; do
  room
  echo "== v3 $a" >> $L
  timeout 30m python scripts/v11/v11_imgpath.py $a >> $L 2>&1 || echo "FAILED $a rc=$?" >> $L
done
echo "== done5" >> $L
echo "$(now) END v11b_cmp" >> $Q
rc=1
for n in 0 1 2 3; do
  room
  before=$(wc -l < logs/watchdog_g1.stopped)
  echo "$(now) START v11b_noncam2 (try $n)" >> $Q
  setsid timeout 300m python scripts/v11/v11b_noncam.py --out $O >> logs/v11b_noncam2.log 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 30
    [ -e logs/watchdog_g1.paused ] && touch logs/v11b_noncam2.log  # a paused job is not a stalled one
    age=$(( $(date +%s) - $(stat -c %Y logs/v11b_noncam2.log) ))
    if [ $age -gt 2400 ]; then echo "$(now) STALL v11b_noncam2 (${age}s)" >> $Q; kill -- -$pid; sleep 15; kill -9 -- -$pid; fi
  done
  wait $pid; rc=$?
  echo "$(now) END v11b_noncam2 rc=$rc" >> $Q
  [ $rc = 0 ] && break
  [ "$(wc -l < logs/watchdog_g1.stopped)" -gt "$before" ] || break
  sleep 600
done
python scripts/v11/v11b_data_check.py $O > logs/v11b_data_check2.log 2>&1
echo "$(now) QUEUE END v11b_q3 noncam=$rc check=$?" >> $Q
