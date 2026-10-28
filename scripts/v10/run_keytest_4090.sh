#!/bin/bash
# Reference-pick offline check on the AutoDL 4090 (10-23): v10_key_pca.py (render + pi0.5 image tokens, EGL + CUDA on
# GPU 0) then v10_key_film.py (raw / PCA / learned / FiLM keys). Logs logs/v10_key_{pca,film}.log, KEYTEST_DONE at the end.
cd /root/pace
source /root/pace/env_plus_4090.sh
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/PACE/code_v10:/root/pace/scripts/v10
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2
python scripts/v10/v10_key_pca.py --workers 6 --threads 8 --device cuda > logs/v10_key_pca.log 2>&1
rc=$?; echo "[$(date "+%F %T")] KEYPCA_DONE rc=$rc" >> logs/v10_key_pca.log
[ $rc = 0 ] && OMP_NUM_THREADS=32 python scripts/v10/v10_key_film.py > logs/v10_key_film.log 2>&1
rc=$?; echo "[$(date "+%F %T")] KEYTEST_DONE rc=$rc" >> logs/v10_key_film.log
