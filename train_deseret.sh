CUDA_VISIBLE_DEVICES="" python train.py \
    --model="microsoft/trocr-base-stage1" \
    --dataset="cayjobla/iam-deseret" \
    --early_stopping \
    --logging_steps=1 \
    --save_steps=100 \
    --eval_steps=100 \
    --output_dir="trocr-base-deseret-test"

