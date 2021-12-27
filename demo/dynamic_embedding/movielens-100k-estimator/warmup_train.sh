#!/usr/bin/env bash
sh stop.sh

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "chief", "index": 0}}'
python movielens-100k-estimator.py --mode warmup &
sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "worker", "index": 0}}'
python movielens-100k-estimator.py --mode warmup &
sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "ps", "index": 0}}'
python movielens-100k-estimator.py --mode warmup &

echo "ok"
