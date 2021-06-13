#!/usr/bin/env bash

PARTITION="spring_scheduler"
JOB_NAME=$1
GPUS=$2
CONFIG=$3

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=5

PROT=18888

srun -p ${PARTITION} \
    --comment=spring-submit \
    --job-name=${JOB_NAME} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=1 \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
python ../../tools/train_val.py \
    --config ${CONFIG} \
    # --e \
