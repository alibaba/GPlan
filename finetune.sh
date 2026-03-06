#!/bin/bash
# Progressive Implicit CoT Distillation Finetune Script
# Distributed finetuning with DeepSpeed ZeRO-3

torchrun --nproc_per_node=4 finetune.py \
  --resume_from_checkpoint=... \
  --train_csv=./data_process/dataset/train.csv \
  --output_dir=./output \
  --extended_tokens=./add_tokens/extended_cot_vocabs.json \
  --cot_mode=progressive_cot_distill \
  --epochs=10 \
  --per_device_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-5 \
  --model_max_length=2048 \
  --cot_weight=1.0 \
  --json_weight=1.0 \
  --distill_direction=forward \
  --deepspeed=./config/ds_z3_bf16.json \
  --bf16 \
  --logging_step=10 \
  --warmup_ratio=0.01
