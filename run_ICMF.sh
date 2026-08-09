#!/bin/bash

HNC_MARGIN=0.2
HNC_TOPK=3
DROPOUT=0.1

CUDA_VISIBLE_DEVICES=$1 python -u main.py \
    --data_choice   $2 \
    --data_split    $3 \
    --data_rate     $4 \
    --lr            5e-4 \
    --dropout       $DROPOUT \
    --attn_dropout  $DROPOUT \
    --epoch         500 \
    --optimizer_reset_interval 50 \
    --batch_size    3500 \
    --eval_epoch    2 \
    --workers       4 \
    --csls \
    --use_dvdc      1 \
    --dvdc_target_disp 0.1 \
    --dvdc_lr       0.05 \
    --dvdc_ema_momentum 0.95 \
    --use_hnc       1 \
    --hnc_margin    $HNC_MARGIN \
    --hnc_topk      $HNC_TOPK \
    --disable_early_stop 1 \
    --early_stop_metric mrr_l2r \
    --save_model    1 \
    --exp_name      ICMF_${2}_${3}_${4}_NI_HNC \
    --exp_id        v1
