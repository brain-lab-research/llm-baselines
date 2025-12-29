#!/bin/bash
# torchrun --nproc_per_node=2 --master_port=1234
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=4

# python ./src/optim/lipschitz_analyzer.py \
# python ./src/main.py \

python ./src/optim/lipschitz_analyzer.py \
    --model llama \
    --dataset fineweb \
    --optimizer signum \
    --lr 1e-4 \
    --div_factor 100 \
    --iterations 10000 \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 512 \
    --acc_steps 1 \
    --warmup_steps 1000 \
    --grad_clip 0.5 \
    --seed 0 \
    --weight_decay 0.1 \
    --scheduler cos \
    --momentum 0.9 \
    --dropout 0 \
    --beta1 0.8 --beta2 0.999 \
    --dropout 0.0 \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --analyze_lipschitz \
    --weight_norm_type signum \
    --rho 2 \
    --f_star 3.2 \
    --log_interval 1 \
    --do_not_auto_resume \
    --wandb # \ --do_not_auto_resume \ --fit_rho \
