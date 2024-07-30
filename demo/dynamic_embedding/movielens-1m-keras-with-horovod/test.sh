#!/bin/bash

python movielens-1m-keras-with-horovod.py --mode=${1:-"test"} --export_dir="./export_dir" --shuffle=${2:-False}