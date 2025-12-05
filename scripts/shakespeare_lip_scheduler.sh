#!/bin/bash

source ./optim_venv/bin/activate

export CUDA_VISIBLE_DEVICES=3
python ./src/main.py \
    --model base \
    --dataset shakespeare-char \
    --opt muon \
    --lr 1e-1 \
    --momentum 0.9 \
    --iterations 1000 \
    --vocab_size 96 \
    --n_embd 128 \
    --n_head 4 \
    --n_layer 2 \
    --batch_size 64 \
    --sequence_length 128 \
    --acc_steps 1 \
    --warmup_steps 0 \
    --grad_clip 0 \
    --seed 0 \
    --weight_decay 1e-3 \
    --scheduler lipschitz \
    --lipschitz_K_0_via_lr \
    --lipschitz_K_rho 1 \
    --lipschitz_rho 2.0 \
    --lipschitz_loss_star 0.0 \
    --beta1 0.9 --beta2 0.95 \
    --dropout 0.0 \
    --log_interval 1 \
    --eval_interval 10 \
    --wandb
