#!/usr/bin/env bash
# original experiments
CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-squad small_squad.json test_squad.json > xdomain_qa_small_squad.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh large-qamr large_qamr.json test_qamr.json > xdomain_qa_large_qamr.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh large-qasrl large_qasrl.json test_qasrl.json > xdomain_qa_large_qasrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments.sh large-qare large_qare.json test_qare.json > xdomain_qa_large_qare.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh large-newsqa large_newsqa.json test_newsqa.json > xdomain_qa_large_newsqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments.sh large-triviaqa large_triviaqa.json test_triviaqa.json > xdomain_qa_large_triviaqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments.sh small-large-qamr small_large_qamr.json test_squad.json > xdomain_qa_small_large_qamr.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments.sh small-large-qasrl small_large_qasrl.json test_squad.json > xdomain_qa_small_large_qasrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments.sh small-large-qare small_large_qare.json test_squad.json > xdomain_qa_small_large_qare.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-large-newsqa small_large_newsqa.json test_squad.json > xdomain_qa_small_large_newsqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh small-large-triviaqa small_large_triviaqa.json test_squad.json > xdomain_qa_small_large_triviaqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments.sh small-large-squad small_large_squad.json test_squad.json > xdomain_qa_small_large_squad.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_test.sh large-qamr test_squad.json > xdomain_qa_large_qamr_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_test.sh large-qasrl test_squad.json > xdomain_qa_large_qasrl_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_test.sh large-qare test_squad.json > xdomain_qa_large_qare_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_test.sh large-newsqa test_squad.json > xdomain_qa_large_newsqa_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_test.sh large-triviaqa test_squad.json > xdomain_qa_large_triviaqa_test.log 2>&1 &

# experiments with squad as the main task
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh small-squad-2nd small_squad.json test_squad.json > xdomain_qa_small_squad_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh large-qamr-2nd large_qamr.json test_qamr.json > xdomain_qa_large_qamr_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments.sh large-qasrl-2nd large_qasrl.json test_qasrl.json > xdomain_qa_large_qasrl_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments.sh large-qare-2nd large_qare.json test_qare.json > xdomain_qa_large_qare_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments.sh large-newsqa-2nd large_newsqa.json test_newsqa.json > xdomain_qa_large_newsqa_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments.sh large-triviaqa-2nd large_triviaqa.json test_triviaqa.json > xdomain_qa_large_triviaqa_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh large-squad-2nd large_squad.json test_squad.json > xdomain_qa_large_squad_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-large-qamr-2nd small_large_qamr.json test_squad.json > xdomain_qa_small_large_qamr_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh small-large-qasrl-2nd small_large_qasrl.json test_squad.json > xdomain_qa_small_large_qasrl_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments.sh small-large-qare-2nd small_large_qare.json test_squad.json > xdomain_qa_small_large_qare_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh small-large-newsqa-2nd small_large_newsqa.json test_squad.json > xdomain_qa_small_large_newsqa_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-large-triviaqa-2nd small_large_triviaqa.json test_squad.json > xdomain_qa_small_large_triviaqa_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh small-large-squad-2nd small_large_squad.json test_squad.json > xdomain_qa_small_large_squad_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_test.sh large-qamr-2nd test_squad.json > xdomain_qa_large_qamr_test_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_test.sh large-qasrl-2nd test_squad.json > xdomain_qa_large_qasrl_test_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_test.sh large-qare-2nd test_squad.json > xdomain_qa_large_qare_test_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_test.sh large-newsqa-2nd test_squad.json > xdomain_qa_large_newsqa_test_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_test.sh large-triviaqa-2nd test_squad.json > xdomain_qa_large_triviaqa_test_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_pretraining.sh small-large-qamr-pretraining-2nd large-qamr-2nd small_squad.json test_squad.json > xdomain_qa_small_large_qamr_pretraining_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_pretraining.sh small-large-qasrl-pretraining-2nd large-qasrl-2nd small_squad.json test_squad.json > xdomain_qa_small_large_qasrl_pretraining_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_pretraining.sh small-large-qare-pretraining-2nd large-qare-2nd small_squad.json test_squad.json > xdomain_qa_small_large_qare_pretraining_2nd.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_pretraining.sh small-large-newsqa-pretraining-2nd large-newsqa-2nd small_squad.json test_squad.json > xdomain_qa_small_large_newsqa_pretraining_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_pretraining.sh small-large-triviaqa-pretraining-2nd large-triviaqa-2nd small_squad.json test_squad.json > xdomain_qa_small_large_triviaqa_pretraining_2nd.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_pretraining.sh small-large-squad-pretraining-2nd large-squad-2nd small_squad.json test_squad.json > xdomain_qa_small_large_squad_pretraining_2nd.log 2>&1 &

# experiments with qamr as the main task
CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-qamr small_qamr.json test_qamr.json > xdomain_qa_qamr_small_qamr.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh large-qamr large_qamr.json test_qamr.json > xdomain_qa_qamr_large_qamr.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments.sh large-qasrl large_qasrl.json test_qasrl.json > xdomain_qa_qamr_large_qasrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh large-qare large_qare.json test_qare.json > xdomain_qa_qamr_large_qare.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments.sh large-newsqa large_newsqa.json test_newsqa.json > xdomain_qa_qamr_large_newsqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments.sh large-triviaqa large_triviaqa.json test_triviaqa.json > xdomain_qa_qamr_large_triviaqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments.sh large-squad large_squad.json test_squad.json > xdomain_qa_qamr_large_squad.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments.sh small-large-qamr small_large_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_qamr.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh small-large-qasrl small_large_qasrl.json test_qamr.json > xdomain_qa_qamr_small_large_qasrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments.sh small-large-qare small_large_qare.json test_qamr.json > xdomain_qa_qamr_small_large_qare.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments.sh small-large-newsqa small_large_newsqa.json test_qamr.json > xdomain_qa_qamr_small_large_newsqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments.sh small-large-triviaqa small_large_triviaqa.json test_qamr.json > xdomain_qa_qamr_small_large_triviaqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments.sh small-large-squad small_large_squad.json test_qamr.json > xdomain_qa_qamr_small_large_squad.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_pretraining.sh small-large-qamr-pretraining large-qamr small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_qamr_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_pretraining.sh small-large-qasrl-pretraining large-qasrl small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_qasrl_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_pretraining.sh small-large-qare-pretraining large-qare small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_qare_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments_pretraining.sh small-large-newsqa-pretraining large-newsqa small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_newsqa_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_pretraining.sh small-large-triviaqa-pretraining large-triviaqa small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_triviaqa_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments_pretraining.sh small-large-squad-pretraining large-squad small_qamr.json test_qamr.json > xdomain_qa_qamr_small_large_squad_pretraining.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_test.sh large-qasrl test_qamr.json > xdomain_qa_qamr_large_qasrl_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_test.sh large-qare test_qamr.json > xdomain_qa_qamr_large_qare_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_test.sh large-newsqa test_qamr.json > xdomain_qa_qamr_large_newsqa_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_test.sh large-triviaqa test_qamr.json > xdomain_qa_qamr_large_triviaqa_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_test.sh large-squad test_qamr.json > xdomain_qa_qamr_large_squad_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments_test.sh small-large-qamr-pretraining test_qamr.json > xdomain_qa_qamr_small_large_qamr_pretraining_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_test.sh small-large-qasrl-pretraining test_qamr.json > xdomain_qa_qamr_small_large_qasrl_pretraining_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_test.sh small-large-qare-pretraining test_qamr.json > xdomain_qa_qamr_small_large_qare_pretraining_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_test.sh small-large-newsqa-pretraining test_qamr.json > xdomain_qa_qamr_small_large_newsqa_pretraining_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_test.sh small-large-triviaqa-pretraining test_qamr.json > xdomain_qa_qamr_small_large_triviaqa_pretraining_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments_test.sh small-large-squad-pretraining test_qamr.json > xdomain_qa_qamr_small_large_squad_pretraining_test.log 2>&1 &

# experiments with newsqa as the main task
CUDA_VISIBLE_DEVICES=0 nohup sh run_qa_experiments.sh small-newsqa small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_newsqa.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments.sh small-large-qamr small_large_qamr.json test_newsqa.json > xdomain_qa_newsqa_small_large_qamr.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments.sh small-large-qasrl small_large_qasrl.json test_newsqa.json > xdomain_qa_newsqa_small_large_qasrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments.sh small-large-qare small_large_qare.json test_newsqa.json > xdomain_qa_newsqa_small_large_qare.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments.sh small-large-newsqa small_large_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_newsqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments.sh small-large-triviaqa small_large_triviaqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_triviaqa.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments.sh small-large-squad small_large_squad.json test_newsqa.json > xdomain_qa_newsqa_small_large_squad.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup sh run_qa_experiments_pretraining.sh small-large-qamr-pretraining large-qamr small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_qamr_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_pretraining.sh small-large-qasrl-pretraining large-qasrl small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_qasrl_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_pretraining.sh small-large-qare-pretraining large-qare small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_qare_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run_qa_experiments_pretraining.sh small-large-newsqa-pretraining large-newsqa small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_newsqa_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup sh run_qa_experiments_pretraining.sh small-large-triviaqa-pretraining large-triviaqa small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_triviaqa_pretraining.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run_qa_experiments_pretraining.sh small-large-squad-pretraining large-squad small_newsqa.json test_newsqa.json > xdomain_qa_newsqa_small_large_squad_pretraining.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup sh run_qa_experiments_test.sh large-qamr test_newsqa.json > xdomain_qa_newsqa_large_qamr_test.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run_qa_experiments_test.sh large-qasrl test_newsqa.json > xdomain_qa_newsqa_large_qasrl_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments_test.sh large-qare test_newsqa.json > xdomain_qa_newsqa_large_qare_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments_test.sh large-triviaqa test_newsqa.json > xdomain_qa_newsqa_large_triviaqa_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup sh run_qa_experiments_test.sh large-squad test_newsqa.json > xdomain_qa_newsqa_large_squad_test.log 2>&1 &


