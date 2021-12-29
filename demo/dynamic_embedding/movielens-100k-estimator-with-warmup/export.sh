#!/usr/bin/env bash

rm -rf ./export_dir
sh stop.sh

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "chief", "index": 0}}'
python movielens-100k-estimator.py --mode serving &

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "worker", "index": 0}}'
python movielens-100k-estimator.py --mode serving &

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "ps", "index": 0}}'
python movielens-100k-estimator.py --mode serving &

echo "ok"
