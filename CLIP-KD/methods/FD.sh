#!/usr/bin/env bash

cd /home/brownieewha/CLIP-KD/src
torchrun --nproc_per_node 2 -m \
    --master_addr=127.0.0.1 --master_port=29534 \
    training.main_kd -- \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/acpl-ssd30/brownieewha/data/cc3m/cc3m_train.csv"  \
    --val-data="/acpl-ssd30/brownieewha/data/cc3m/cc3m_val.csv"  \
    --data-root /acpl-ssd30/brownieewha/data/cc3m/images/ \
    --val-data-root /acpl-ssd30/brownieewha/data/cc3m/images/ \
    --csv-img-key filepath \
    --csv-caption-key title \
    --imagenet-val=/acpl-ssd30/brownieewha/data/imageNet_val \
    --warmup 10000 \
    --batch-size=512 \
    --lr=1e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model ViT-T-16 \
    --t-model ViT-B-16 \
    --t-model-checkpoint /home/brownieewha/CLIP-KD/checkpoint/ViT_B_16-laion400m_e32.pt \
    --logs /home/brownieewha/CLIP-KD/logs/vit_b_16_method_logs_with_laion400m_checkpoint/ \
    --alpha_fd_loss 2000. \
    --tag distill-FD-new 
