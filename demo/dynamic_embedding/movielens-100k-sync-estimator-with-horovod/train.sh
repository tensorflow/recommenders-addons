#!/usr/bin/env bash
rm -rf ./ckpt
rm -rf ./export_dir
sh stop.sh

sleep 1
export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "chief": ["localhost:2228"], "worker":["localhost:2240", "localhost:2241"]}, "task": {"type": "ps", "index": 0}}'
python movielens-100k-estimator.py --mode train &

mpirun -np 2 -H localhost:2 --allow-run-as-root -bind-to none -map-by slot sh -c './start_worker.sh > log/worker_$OMPI_COMM_WORLD_RANK.log 2>&1' 

echo "ok"
