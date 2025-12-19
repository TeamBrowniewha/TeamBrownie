cd /home/brownieewha/CLIP-KD/src
CUDA_VISIBLE_DEVICES="" \
python -m training.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --zeroshot=/home/brownieewha/CLIP-KD/zeroshot_data/tv_exports/flowers102_test/test/test \
    --batch-size=2048 \
    --epochs 32 \
    --workers=16 \
    --model ViT-T-16 \
    --resume /home/brownieewha/CLIP-KD/checkpoint/vit_t_16_method_ckpt_with_cc3m+12m/method_vit_t_16_crd+icl+fd.pt \
    --logs /home/brownieewha/CLIP-KD/logs/zeroshot/vit_t_16_methods_with_cc3m+12m/vit_t_16_method_crd+icl+fd \
    --eval \
    --tag vit-t-16-method_with_cc3m+12m_crd+icl+fd-Flowers102