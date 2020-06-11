#!/usr/bin/env bash
echo "data option: " $1
echo "test file: " $2
export DATA_DIR=/path/to/working/dir/data/xdomain-QA-newsqa
export OUTPUT_DIR=/path/to/working/dir/models/xdomain-QA-newsqa/$1

python run_squad.py \
  --model_type bert \
  --model_name_or_path $OUTPUT_DIR \
  --do_eval \
  --predict_file $DATA_DIR/$2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR