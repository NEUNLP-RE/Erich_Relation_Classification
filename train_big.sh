#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

code_dir=./src
dataset=./dataset/semeval
save_dir=./checkpoints/semeval
pretrain_dir=./pretrain_model/uncased_L-24_H-1024_A-16/
result_file=./dataset/semeval/eval_result.txt


wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip
unzip uncased_L-24_H-1024_A-16.zip -d $pretrain_dir
mv $pretrain_dir/uncased_L-24_H-1024_A-16/bert_config.json $pretrain_dir/uncased_L-24_H-1024_A-16/config.json


python scripts/convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path $pretrain_dir/uncased_L-24_H-1024_A-16/bert_model.ckpt \
    --bert_config_file $pretrain_dir/uncased_L-24_H-1024_A-16/config.json \
    --pytorch_dump_path $pretrain_dir/uncased_L-24_H-1024_A-16/pytorch_model.bin


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
    --fp16
