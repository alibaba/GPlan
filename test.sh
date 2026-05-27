#!/bin/bash
# GPlan Open Test Script
# Single-GPU or multi-GPU inference with evaluation metrics
# Metrics: Acc@1, NDCG@3, NES

torchrun --nproc_per_node=4 test.py \
  --resume_from_checkpoint=./output \
  --valid_csv=./data_process/dataset/test.csv \
  --output_dir=./test_results \
  --cot_mode=latent_multi_cot \
  --model_max_length=2048 \
  --test_batch_size=4 \
  --tool_mismatch_cost=1.0 \
  --param_mismatch_cost=0.3 \
  --bf16
