#!/bin/bash

data_file=./raw-text/financial/1014.xlsx
datatype=v1
task_name=financial
raw_data=./dataset
data_dir=${raw_data}/${task_name}/${datatype}
static_dir=$data_dir/static

if [ ! -d $data_dir ]; then
  mkdir -p $data_dir
  mkdir -p $static_dir
else
  rm -r $data_dir
  echo "dir exist!!"
  exit
fi

python scripts/produce_relations_by_excel.py -e $data_file -r $static_dir/all_relation2id.txt
python scripts/parser_excel.py -e $data_file -r $static_dir/all_relation2id.txt -o $data_dir/output.txt
python scripts/remove_repeat.py $data_dir/output.txt $data_dir/norep_output.txt

python scripts/count_relations.py \
    -t $data_dir/norep_output.txt \
    -r $static_dir/all_relation2id.txt \
    -s $static_dir/static_output.txt \
    -n $static_dir/nodir_static_output.txt

python scripts/produce_top_relations.py \
    -n $static_dir/nodir_static_output.txt \
    -d $static_dir/static_output.txt \
    -t 40 \
    -r $static_dir/new_relation2id.txt \
    -c $static_dir/new_nodir_static_output.txt \
    -s $static_dir/new_static_output.txt

cp $static_dir/new_relation2id.txt $data_dir/relation2id.txt

python scripts/convert_output_all2topn.py \
    -i $data_dir/norep_output.txt \
    -o $static_dir/all_relation2id.txt \
    -r $data_dir/relation2id.txt \
    -n $data_dir/new_output.txt

python scripts/split_test_train.txt.py \
    -s $data_dir/new_output.txt \
    -r $data_dir/relation2id.txt \
    -c $data_dir/test.txt \
    -x $data_dir/train.txt

#rm -r $data_dir

