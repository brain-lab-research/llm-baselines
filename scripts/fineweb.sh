#!/bin/bash
# torchrun --nproc_per_node=2 --master_port=1234
# --distributed_backend nccl \
export CUDA_VISIBLE_DEVICES=0

python ./src/main.py \
    --model llama \
    --dataset fineweb \
    --optimizer muon \
    --lr 5e-6 \
    --iterations 7000 \
    --n_embd 768 \
    --n_head 12 \
    --n_layer 12 \
    --batch_size 64 \
    --sequence_length 512 \
    --acc_steps 1 \
    --warmup_steps 500 \
    --grad_clip 0 \
    --seed 0 \
    --weight_decay 0.1 \
    --scheduler none \
    --div_factor 100 \
    --final_div_factor 0.1 \
    --momentum 0.9 \
    --beta1 0.9 --beta2 0.95 \
    --dropout 0.0 \
    --eval_interval 115 --latest_ckpt_interval 1000 \
    --analyze_lipschitz \
    --min_analysis_steps 100 \
    --weight_norm_type frobenius \
    --rho 2 \
    --f_star 3.2 \
    --log_interval 1 \
    --do_not_auto_resume \
    --wandb # \ --do_not_auto_resume \ --fit_rho \
