#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 nohup sh run_ner_experiments.sh small-twitter small_twitter.txt test_twitter.txt test_twitter.txt > xdomain_small_twitter.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_ner_experiments.sh small-large-twitter small_large_twitter.txt test_twitter.txt test_twitter.txt > xdomain_small_large_twitter.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_ner_experiments.sh small-large-ontonotes small_large_ontonotes.txt test_ontonotes.txt test_twitter.txt > xdomain_small_large_ontonotes.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_ner_experiments.sh small-large-conll small_large_conll.txt test_conll.txt test_twitter.txt > xdomain_small_large_conll.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_ner_experiments.sh small-large-GMB small_large_GMB.txt test_GMB.txt test_twitter.txt > xdomain_small_large_GMB.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_ner_experiments.sh large-twitter large_twitter.txt test_twitter.txt test_twitter.txt > xdomain_large_twitter.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_ner_experiments.sh large-ontonotes large_ontonotes.txt test_ontonotes.txt test_twitter.txt > xdomain_large_ontonotes.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_ner_experiments.sh large-conll large_conll.txt test_conll.txt test_twitter.txt > xdomain_large_conll.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_ner_experiments.sh large-GMB large_GMB.txt test_GMB.txt test_twitter.txt > xdomain_large_GMB.log 2>&1 &

# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-ontonotes test_ontonotes.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-ontonotes test_conll.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-ontonotes test_twitter.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-ontonotes test_GMB.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-ontonotes test_ontonotes.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-ontonotes test_conll.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-ontonotes test_twitter.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-ontonotes test_GMB.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-conll test_ontonotes.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-conll test_conll.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-conll test_twitter.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-conll test_GMB.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-twitter test_ontonotes.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-twitter test_conll.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-twitter test_twitter.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-twitter test_GMB.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-GMB test_ontonotes.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-GMB test_conll.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-GMB test_twitter.txt
# CUDA_VISIBLE_DEVICES=0 sh run_ner_experiments_test.sh small-large-GMB test_GMB.txt
