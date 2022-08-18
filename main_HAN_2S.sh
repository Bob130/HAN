#!/usr/bin/env bash

# Gesture

name=HAN_2S_1e-3


CUDA_VISIBLE_DEVICES=1  \
python main_HAN_2S.py \
--checkpoint checkpoint/SHREC/${name}/ \
--dpout_rate 0.1 \
--dataset HandGesture \
--class_num 14 \
--workers 4 \
--input_frames 8 \
--raw_input_dim 3 \
--lr 1e-3 \
--weight_decay 0 \
--momentum 0 \
--epochs 1000 \
--reduce_lr_on_plateau \
--gamma 0.1 \
--patience 50 \
--seed -1 \
--loop 1 \
--train_batch 32 \
--test_batch 32 \
--num_gpu 1 \
#-e







