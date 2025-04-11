#!/bin/bash
rm -rf ./ckpt
sh stop.sh
sleep 1
python movielens-1m-keras-ps.py --ps_list="localhost:2220" --worker_list="localhost:2231" --chief="localhost:2230" --task_mode="ps" --task_id=0 &
sleep 1
python movielens-1m-keras-ps.py --ps_list="localhost:2220" --worker_list="localhost:2231" --chief="localhost:2230" --task_mode="worker" --task_id=0 &
sleep 1
python movielens-1m-keras-ps.py --ps_list="localhost:2220" --worker_list="localhost:2231" --chief="localhost:2230" --task_mode="chief" --task_id=0
echo "ok"