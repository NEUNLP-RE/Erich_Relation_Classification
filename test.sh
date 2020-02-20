#!/bin/bash

orig=./SemEval2010_task8_all_data
dataset=./dataset/semeval
score_dir=$dataset/semeval_score
result_file=$dataset/eval_result.txt

if [ ! -d $score_dir ]; then
    mkdir -p $score_dir
fi

python scripts/convert_rel2id.py \
    -i $result_file \
    -r $dataset/relation2id.txt \
    -t $score_dir/sem_res.txt

# python scripts/extract_test_key.py \
#     -s $orig/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT \
#     -k $score_dir/test_key.txt
./scripts/semeval2010_task8_scorer-v1.2.pl $score_dir/sem_res.txt $score_dir/test_key.txt

