#!/bin/bash

# check whether model path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi


MODEL_PATH=$1

# generate 8 server instances
for i in {0..7}
do
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.api_server \
        --model $MODEL_PATH \
        --gpu-memory-utilization=0.9 \
        --max-num-seqs=200 \
        --host 127.0.0.1 --tensor-parallel-size 1 \
        --port $((8000+i)) \
    &
done
