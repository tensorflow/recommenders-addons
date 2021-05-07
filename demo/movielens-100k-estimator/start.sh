#!/bin/bash
rm -rf ./ckpt
sh stop.sh
sleep 1
python movielens-100k-estimator.py --ps_list="localhost:2220,localhost:2221" --worker_list="localhost:2230,localhost:2231" --task_mode="ps" --task_id=0 &
sleep 1
python movielens-100k-estimator.py --ps_list="localhost:2220,localhost:2221" --worker_list="localhost:2230,localhost:2231" --task_mode="ps" --task_id=1 &
sleep 1
python movielens-100k-estimator.py --ps_list="localhost:2220,localhost:2221" --worker_list="localhost:2230,localhost:2231" --task_mode="worker" --task_id=1 --is_chief=False &
sleep 1
python movielens-100k-estimator.py --ps_list="localhost:2220,localhost:2221" --worker_list="localhost:2230,localhost:2231" --task_mode="worker" --task_id=0 --is_chief=True &
echo "ok"