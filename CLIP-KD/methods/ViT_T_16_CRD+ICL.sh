#!/usr/bin/env bash

cd /home/brownieewha/CLIP-KD/src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.1 --master_port=29534 \
    training.main_kd -- \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/home/brownieewha/CLIP-KD/data/cc3m/cc3m_train.csv"  \
    --val-data="/home/brownieewha/CLIP-KD/data/cc3m/cc3m_val.csv"  \
    --data-root /home/brownieewha/CLIP-KD/data/cc3m/images/images/ \
    --val-data-root /home/brownieewha/CLIP-KD/data/cc3m/images/images/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/home/brownieewha/CLIP-KD/data/imageNet_val \
    --warmup 10000 \
    --batch-size=512 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model ViT-T-16 \
    --t-model ViT-B-16 \
    --t-model-checkpoint /home/brownieewha/CLIP-KD/checkpoint/ViT_B_16_cc3m+12m_checkpoint_final.pt \
    --logs /home/brownieewha/CLIP-KD/logs/ViT_T_16_method_logs_with_cc3m+12m_checkpoint/vit_t_16_CRD+ICL/ \
    --alpha_ckd_loss 1. \
    --alpha_icl_loss 1. \
    --tag distill-new
