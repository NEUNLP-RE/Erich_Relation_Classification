#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

code_dir=./src
dataset=./dataset/semeval
save_dir=./checkpoints/semeval_base
pretrain_dir=./pretrain_model/uncased_L-12_H-768_A-12/
result_file=./dataset/semeval/eval_result.txt
tensorboard_dir=./runs/semeval_base

python $code_dir/main.py \
    --model_type eirc \
    --model_name_or_path $pretrain_dir \
    --task_name semeval \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --data_dir $dataset \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_train_batch_size=16 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --output_dir $save_dir \
    --logging_steps -1 \
    --save_steps -1 \
    --result_file $result_file \
    --tensorboard_dir $tensorboard_dir
