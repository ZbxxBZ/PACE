# 11-01 (user: "你做一个队列自动接续，用gpu1来做吧"; GPU1, container pace_g1; never push out other users' jobs). Three lanes:
#  out  : existing V11 on the remaining goal-out-of-frame camera variants (confirmation of the post-hoc split)
#  iso  : baseline probe of run2, smoke, short training with ref_isolation off, probe
#  orig : references with the camera in place, short training with ref_isolation off on them, probe
# Short trainings: 3000 steps, otherwise run2's settings. Criteria registered in desktop doc 10.14f before the start.
cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1 \
  PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b
Q=logs/queue_run.log
CK=runs/v11/train/v9_run2_4090/best.pt
T=runs/v11/closed/outframe/targets.json
now() { date "+%m-%d %H:%M:%S"; }
room() {  # GPU GB this stage needs: wait while GPU1 used + need > 44 GB
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ $((used + $1 * 1000)) -le 44000 ] && break
    echo "$(now) $2 waits: GPU1 ${used} MiB used" >> $Q; sleep 300
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
    age=$(( $(date +%s) - $(stat -c %Y $log) ))
    if [ $age -gt $stall ]; then echo "$(now) STALL $name (${age}s)" >> $Q; kill -- -$pid; sleep 15; kill -9 -- -$pid; fi
  done
  wait $pid; local rc=$?
  echo "$(now) END $name rc=$rc" >> $Q
  return $rc
}
A="--ckpt $CK --refs_dir runs/v11/closed/run1 --targets $T --inits 1 2 3 --one_per_cell --modes v11 v11other base"
TR="--steps 3000 --bs 16 --accum 2 --workers 4 --w_exec 1000 --ref_isolation off"
lane_out() {
  stage v11b_out_smoke 12 60 1800 python scripts/v11/closed_loop_v11.py $A --limit 2 --batch 2 --out runs/v11/closed/outframe_smoke || return 1
  stage v11b_out 12 500 2400 python scripts/v11/closed_loop_v11.py $A --batch 8 --out runs/v11/closed/outframe
}
lane_iso() {
  stage v11b_probe_run2 12 30 900 python scripts/v11/v11_paths.py runs/v11/train/v9_run2_4090 best.pt runs/v11/probe/run2.json || return 1
  stage v11b_smoke 14 30 900 python scripts/v11/train_v11_v9.py --out runs/v11/train/b_smoke --smoke --bs 8 --accum 1 --workers 2 --w_exec 1000 --ref_isolation off || return 1
  stage v11b_iso 16 400 1800 python scripts/v11/train_v11_v9.py --out runs/v11/train/b_iso $TR || return 1
  stage v11b_iso_probe 12 30 900 python scripts/v11/v11_paths.py runs/v11/train/b_iso last.pt runs/v11/probe/b_iso.json
}
lane_orig() {
  stage v11b_refvis 10 180 1800 python scripts/v11/v11_refvis_orig.py --out /root/pace/runs/v11/refvis_orig || return 1
  stage v11b_orig 16 400 1800 python scripts/v11/train_v11_v9.py --out runs/v11/train/b_iso_orig $TR --refvis /root/pace/runs/v11/refvis_orig || return 1
  stage v11b_orig_probe 12 30 900 python scripts/v11/v11_paths.py runs/v11/train/b_iso_orig last.pt runs/v11/probe/b_iso_orig.json
}
echo "$(now) QUEUE START v11b_q1" >> $Q
lane_out & a=$!
lane_iso & b=$!
lane_orig & c=$!
wait $a; ra=$?; wait $b; rb=$?; wait $c; rc=$?
echo "$(now) QUEUE END v11b_q1 out=$ra iso=$rb orig=$rc" >> $Q
