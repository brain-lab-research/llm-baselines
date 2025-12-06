#!/bin/bash
# torchrun --nproc_per_node=2 --master_port=1234
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=6

python ./src/main.py \
    --model llama \
    --dataset fineweb \
    --optimizer muon \
    --lr 1e-3 \
    --div_factor 25.0 \
    --iterations 64000 \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 512 \
    --acc_steps 1 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay 0.1 \
    --scheduler lipschitz \
    --lipschitz_rho 2.0 \
    --lipschitz_mode intergral_0_x0 \
    --lipschitz_use_cos \
    --lipschitz_loss_star 0 \
    --momentum 0.9 \
    --dropout 0.1 \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --log_interval 1 \
    --do_not_auto_resume \
    --wandb # \ --do_not_auto_resume \ --fit_rho \ --warmup_steps 2000 \

# lipschitz_mode: double_prime func_prime intergral_0_x0
# 