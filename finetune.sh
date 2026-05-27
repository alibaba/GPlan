#!/bin/bash
# PICD Finetune Script
# Distributed finetuning with DeepSpeed ZeRO-3
#
# The public GSISR train.csv contains JSON labels only. PICD training requires
# a CoT-augmented CSV whose raw_labels field contains <THOUGHT>...</THOUGHT>
# followed by the JSON plan.

TRAIN_CSV=${TRAIN_CSV:-./path/to/cot_augmented_train.csv}

torchrun --nproc_per_node=4 finetune.py \
  --resume_from_checkpoint=... \
  --train_csv="${TRAIN_CSV}" \
  --output_dir=./output \
  --extended_tokens=./add_tokens/extended_cot_vocabs.json \
  --cot_mode=latent_multi_cot \
  --epochs=13 \
  --per_device_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --model_max_length=2048 \
  --cot_weight=1.0 \
  --json_weight=1.0 \
  --picd_lr_schedule=compression_aware \
  --picd_lr_target_compressed_blocks=9 \
  --picd_polish_lr=1e-6 \
  --deepspeed=./config/ds_z3_bf16.json \
  --bf16 \
  --logging_step=10 \
  --warmup_ratio=0.01
