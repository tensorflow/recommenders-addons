#!/bin/bash
rm -rf ./export_dir
gpu_num=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
export gpu_num
horovodrun -np $gpu_num python movielens-1m-keras-with-horovod.py --mode="train" --model_dir="./model_dir" --export_dir="./export_dir"