#!/bin/bash
rm -rf ./export_dir
gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export gpu_num
export TF_XLA_FLAGS=--tf_xla_auto_jit=2
horovodrun -np $gpu_num python seq_and_dense.py --mode="train" --model_dir="./model_dir" --export_dir="./export_dir" \
  --steps_per_epoch=${1:-20000} --shuffle=${2:-True}