#!/bin/bash

orig=./SemEval2010_task8_all_data
dataset=./dataset/semeval
tokenizer=./tokenizer/stanford-corenlp-full-2018-10-05
pretrain_dir=./pretrain_model

if [ ! -d $dataset ]; then
    mkdir -p $dataset
fi

if [ ! -d $pretrain_dir ]; then
    mkdir -p $pretrain_dir
fi

wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip -d $pretrain_dir
cp $pretrain_dir/uncased_L-12_H-768_A-12/bert_config.json $pretrain_dir/uncased_L-12_H-768_A-12/config.json

<<men
# count relations
python scripts/produce_relation_semeval.py \
    -s $orig/SemEval2010_task8_training/TRAIN_FILE.TXT \
    -r $dataset/relation2id.txt

# produce trian and eval/test
python scripts/prepare_semeval.py \
    -s $orig/SemEval2010_task8_training/TRAIN_FILE.TXT \
    -t $tokenizer \
    -r $dataset/relation2id.txt \
    -f $dataset/train.txt
python scripts/prepare_semeval.py \
    -s $orig/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT \
    -t $tokenizer \
    -r $dataset/relation2id.txt \
    -f $dataset/test.txt
cp $dataset/test.txt $dataset/dev.txt
men


