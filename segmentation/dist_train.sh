#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-29502}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}