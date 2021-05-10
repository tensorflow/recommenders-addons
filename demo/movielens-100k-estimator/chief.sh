#!/usr/bin/env bash

export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "chief", "index": 0}}'
python movielens-100k-estimator.py --mode $1

echo "chief ok"

sleep 100
sh stop.sh
