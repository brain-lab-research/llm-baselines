#!/bin/bash
# torchrun --nproc_per_node=2 --master_port=1234
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=2,3

for f_star in 3.2 3.3 3.4
do
    mkdir -p logs
    torchrun --nproc_per_node=2 --master_port=1222 ./src/main.py \
        --distributed_backend nccl \
        --model llama \
        --dataset fineweb \
        --optimizer d-muon \
        --lr 3e-3    \
        --div_factor 100.0 \
        --lipschitz_sigma_F 0.001 \
        --iterations 16000 \
        --n_embd 768 \
        --n_head 12 \
        --n_layer 24 \
        --batch_size 256 \
        --sequence_length 512 \
        --acc_steps 1 \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay 0.1 \
        --scheduler cos \
        --use_lip_warmup \
        --lipschitz_rho 2.0 \
        --lipschitz_mode linear_and_cos \
        --lipschitz_loss_star $f_star \
        --dropout 0 \
        --beta1 0.8 --beta2 0.999 \
        --eval_interval 115 --latest_ckpt_interval 1000 \
        --log_interval 1000 \
        --do_not_auto_resume > logs/dmuon_fineweb_lip_256_fstar${f_star}.txt 2>&1 \
        #--wandb # \ --do_not_auto_resume \ --fit_rho \ --warmup_steps 2000 \
done