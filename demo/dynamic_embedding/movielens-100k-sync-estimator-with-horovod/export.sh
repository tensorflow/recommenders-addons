#!/usr/bin/env bash
rm -rf ./export_dir
sh stop.sh

sleep 1
export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "chief": ["localhost:2228"], "worker":["localhost:2240"]}, "task": {"type": "ps", "index": 0}}'
python movielens-100k-estimator.py --mode serving &

export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "chief": ["localhost:2228"], "worker":["localhost:2240"]}, "task": {"type": "worker", "index": 0}}'
mpirun -np 1 -H localhost:1 --allow-run-as-root -bind-to none -map-by slot -x TF_CONFIG sh -c 'python movielens-100k-estimator.py --mode serving> log/worker_0.log 2>&1' 

echo "ok"

