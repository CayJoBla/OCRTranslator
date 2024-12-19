#!/bin/bash

#SBATCH --time=4:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32000M   # 32G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --job-name=trocr-steel

export WANDB_PROJECT="cs-674-final-project"
export WANDB_MODE="offline"
python train.py \
    --model="microsoft/trocr-base-printed" \
    --dataset="cayjobla/iam-steel" \
    --max_length=128 \
    --early_stopping \
    --no_repeat_ngram_size=3 \
    --length_penalty=2.0 \
    --num_beams=4 \
    --batch_size=8 \
    --num_epochs=10 \
    --logging_steps=25 \
    --save_steps=2500 \
    --eval_steps=2500 \
    --output_dir="trocr-base-steel" \
    --run_name="trocr-base-steel" \

