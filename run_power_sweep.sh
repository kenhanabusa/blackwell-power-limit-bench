#!/usr/bin/env bash
set -euo pipefail

# conda 環境の有効化
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gpu-bench

# ★ここで GPU メモリまわりの環境変数を設定（手動実行と同じ状態にする）
export CUDA_DEVICE_MAX_CONNECTIONS=1
# 新しい推奨 env
export PYTORCH_ALLOC_CONF=expandable_segments:True
# 互換性のために旧名もつけておく（警告は出ますが効果あり）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/gpu-bench

python run_all_power_sweep.py \
  --train-cmd "torchrun --standalone --nproc_per_node=4 train_ddp_transformer.py --fp16 --seq-len 1280 --global-batch-size 256 --warmup-iters 50 --measure-iters 1000000" \
  --step-seconds 600 \
  --step-watts 100 \
  --log-dir power_sweep_logs \
  --use-sudo
