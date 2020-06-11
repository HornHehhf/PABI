#!/usr/bin/env bash
echo "data option: " $1
echo "test file: " $2

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export DATA_DIR=/path/to/working/dir/data/xdomain-person
export OUTPUT_DIR=/path/to/working/dir/models/xdomain-person/$1

python run_ner.py \
--model_type bert \
--data_dir $DATA_DIR \
--test_file $2 \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict
