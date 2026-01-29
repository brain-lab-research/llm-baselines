#!/bin/bash
# torchrun --nproc_per_node=2 --master_port=1234
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=7

for iter in 64000 128000
do
    for ws in 10 50 100 200 300 400 500
    do
        python ./src/main.py \
            --model llama \
            --dataset fineweb \
            --optimizer normalized-sgd \
            --lr 1e-2 \
            --div_factor 2.0 \
            --iterations $iter \
            --n_embd 768 \
            --n_head 12 \
            --n_layer 24 \
            --batch_size 32 \
            --sequence_length 512 \
            --acc_steps 1 \
            --grad_clip 0.5 \
            --seed 0 \
            --weight_decay 0.1 \
            --scheduler cos \
            --warmup_steps $ws \
            --momentum 0.95 \
            --dropout 0 \
            --beta1 0.9 --beta2 0.99 \
            --eval_interval 115 --latest_ckpt_interval 1000 \
            --log_interval 1 \
            --do_not_auto_resume \
            --wandb # \ --do_not_auto_resume \ --fit_rho \ --warmup_steps 2000 \
    done
done