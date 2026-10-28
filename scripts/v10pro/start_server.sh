#!/bin/bash
# pi0.5 policy server for the LIBERO-PRO clients, on the 4090 (GPU 0). Log: logs/pi05_server.log
source /root/pace/env_plus_4090.sh >/dev/null 2>&1
export HF_HUB_OFFLINE=1 PYTHONPATH=/root/pace/LIBERO-plus:/root/pace/PACE/code_v10:/root/pace/scripts/v10pro CUDA_VISIBLE_DEVICES=0
cd /root/pace/scripts/v10pro
exec python pi05_server.py --port ${PORT:-18765} > /root/pace/logs/pi05_server.log 2>&1
