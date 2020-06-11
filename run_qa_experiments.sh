#!/usr/bin/env bash
echo "data option: " $1
echo "train file: " $2
echo "dev file: " $3
export DATA_DIR=/path/to/working/dir/data/xdomain-QA-newsqa
export OUTPUT_DIR=/path/to/working/dir/models/xdomain-QA-newsqa/$1

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --train_file $DATA_DIR/$2 \
  --predict_file $DATA_DIR/$3 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 5e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir $OUTPUT_DIR