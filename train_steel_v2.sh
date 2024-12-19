#!/bin/bash

#SBATCH --time=12:00:00   # walltime.  hours:minutes:seconds
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=32000M   # 128G memory per CPU core
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --qos=dw87
#SBATCH --partition=dw
#SBATCH --job-name=trocr-steel-v2

export WANDB_PROJECT="cs-674-final-project"
export WANDB_MODE="offline"
python train_steel.py \
    --model="microsoft/trocr-base-stage1" \
    --dataset="Salesforce/wikitext" \
    --dataset_args "wikitext-103-raw-v1" \
    --max_length=64 \
    --early_stopping \
    --no_repeat_ngram_size=3 \
    --length_penalty=2.0 \
    --num_beams=4 \
    --batch_size=8 \
    --num_epochs=1 \
    --logging_steps=100 \
    --save_steps=5000 \
    --eval_steps=5000 \
    --output_dir="trocr-base-steel-wiki" \
    --run_name="trocr-base-steel-wiki" \

