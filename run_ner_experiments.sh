#!/usr/bin/env bash
echo "data option: " $1
echo "train file: " $2
echo "dev file: " $3
echo "test file: " $4

export MAX_LENGTH=256
export BERT_MODEL=bert-base-cased
export DATA_DIR=/path/to/working/dir/data/xdomain-person
export OUTPUT_DIR=/path/to/working/dir/models/xdomain-person/$1
export BATCH_SIZE=8
export NUM_EPOCHS=4
export SEED=1

python run_ner.py \
--model_type bert \
--data_dir $DATA_DIR \
--train_file $2 \
--dev_file $3 \
--test_file $4 \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--seed $SEED \
--do_train \
--do_eval \
--do_predict
