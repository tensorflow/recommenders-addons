#!/usr/bin/env bash
sleep 1
TASK_INEDX=$(($OMPI_COMM_WORLD_RANK))
export TF_CONFIG='{"cluster": {"ps": ["localhost:2223"], "chief": ["localhost:2228"], "worker":["localhost:2240", "localhost:2241"]}, "task": {"type": "worker", "index": '"${TASK_INEDX}"'}}'
python movielens-100k-estimator.py --mode train
