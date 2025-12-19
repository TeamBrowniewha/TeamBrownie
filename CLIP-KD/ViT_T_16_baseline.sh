#!/usr/bin/env bash

cd /home/brownieewha/CLIP-KD/src
torchrun --nproc_per_node 1 -m \
    --master_addr=127.0.0.1 --master_port=29535 \
    training.main -- \
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
    --batch-size=128 \
    --lr=1.25e-3 \
    --wd=0.1 \
    --epochs 32 \
    --workers=8 \
    --model ViT-T-16 \
    --logs /home/brownieewha/CLIP-KD/logs/vit_t_16_baseline/ \
    --tag cc3m-cc12m-baseline
