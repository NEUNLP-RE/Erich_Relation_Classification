#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

code_dir=./src
dataset=./dataset/semeval
# if generating with big model
# save_dir=./checkpoints/semeval_big
save_dir=./checkpoints/semeval_base
result_file=./dataset/semeval/eval_result.txt

python $code_dir/main.py \
    --model_type eirc \
    --model_name_or_path $save_dir \
    --task_name semeval \
    --do_test \
    --do_lower_case \
    --data_dir $dataset \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=16 \
    --output_dir $save_dir \
    --result_file $result_file
