cd /root/pace
source env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/scripts/v11:/root/pace/PACE/code_v10:/root/pace/scripts/v10:/root/pace/scripts/v10b OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=memory.used --format=csv,noheader
timeout 25m python scripts/v11/train_v11b.py --out runs/v11b/train/smoke --align pma --smoke > logs/v11b_smoke.log 2>&1
echo "rc=$?" >> logs/v11b_smoke.log
