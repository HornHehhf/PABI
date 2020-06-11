#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=none > bpp_none.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=labeled > bpp_labeled.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=unlabeled > bpp_unlabeled.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=partial-0.2 > bpp_partial_0.2.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python bpp.py option=partial-0.4 > bpp_partial_0.4.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python bpp.py option=partial-0.6 > bpp_partial_0.6.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python bpp.py option=partial-0.8 > bpp_partial_0.8.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=noisy-0.1 > bpp_noisy_0.1_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=noisy-0.2 > bpp_noisy_0.2_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=noisy-0.3 > bpp_noisy_0.3_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=noisy-0.4 > bpp_noisy_0.4_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python bpp.py option=noisy-0.5 > bpp_noisy_0.5_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python bpp.py option=noisy-0.6 > bpp_noisy_0.6_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python bpp.py option=noisy-0.7 > bpp_noisy_0.7_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=partial+noisy-0.2+0.1 > bpp_partial_noisy_0.2_0.1_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=partial+noisy-0.2+0.2 > bpp_partial_noisy_0.2_0.2_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=partial+noisy-0.2+0.3 > bpp_partial_noisy_0.2_0.3_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=partial+noisy-0.2+0.4 > bpp_partial_noisy_0.2_0.4_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=4 nohup python bpp.py option=partial+noisy-0.4+0.1 > bpp_partial_noisy_0.4_0.1_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=5 nohup python bpp.py option=partial+noisy-0.4+0.2 > bpp_partial_noisy_0.4_0.2_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=6 nohup python bpp.py option=partial+noisy-0.4+0.3 > bpp_partial_noisy_0.4_0.3_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python bpp.py option=partial+noisy-0.4+0.4 > bpp_partial_noisy_0.4_0.4_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=partial+noisy-0.6+0.1 > bpp_partial_noisy_0.6_0.1_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=partial+noisy-0.6+0.2 > bpp_partial_noisy_0.6_0.2_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=partial+noisy-0.6+0.3 > bpp_partial_noisy_0.6_0.3_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=partial+noisy-0.6+0.4 > bpp_partial_noisy_0.6_0.4_1.0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=auxiliary-detection > bpp_auxiliary_detection.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=auxiliary-coarse > bpp_auxiliary_coarse.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=auxiliary-pos > bpp_auxiliary_pos_replace.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=knowledge-1 > bpp_knowledge_1_replace.log 2>&1 &
# CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=knowledge-2 > bpp_knowledge_2_replace.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=constraints-BIO > bpp_constraints_BIO.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=partial+constraints-BIO+0.2 > bpp_partial_constraints_BIO_0.2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=partial+constraints-BIO+0.4 > bpp_partial_constraints_BIO_0.4.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=partial+constraints-BIO+0.6 > bpp_partial_constraints_BIO_0.6.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python bpp.py option=partial+constraints-BIO+0.8 > bpp_partial_constraints_BIO_0.8.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python bpp.py option=constraints-BIO > bpp_constraints_BIO_test_constraint.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python bpp.py option=partial+constraints-BIO+0.2 > bpp_partial_constraints_BIO_0.2_test_constraint.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python bpp.py option=partial+constraints-BIO+0.4 > bpp_partial_constraints_BIO_0.4_test_constraint.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python bpp.py option=partial+constraints-BIO+0.6 > bpp_partial_constraints_BIO_0.6_test_constraint.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python bpp.py option=partial+constraints-BIO+0.8 > bpp_partial_constraints_BIO_0.8_test_constraint.log 2>&1 &
