"""
Progressive Implicit CoT Distillation Finetune Script

Key features:
1. WeightedLossTrainer: Token-level weighted loss with separate cot_loss and json_loss tracking
2. SyncEpochCallback: Sync progressive distillation progress at each epoch
3. WeightedDataCollator: Preserve token_weights and token_types fields during collation
"""

import os
import json
import argparse
import time

import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainerCallback,
)
from dataclasses import dataclass
from typing import Dict, List, Any

from data_process.collate_fns import ProgressiveCotDistillCollater
from data_process.data_loader import CSVDataset
from utils import parse_global_args, parse_train_args, parse_dataset_args, set_seed, ensure_dir


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


class WeightedLossTrainer(transformers.Trainer):
    """
    Custom Trainer with token-level loss weighting.
    Applies different weights to CoT text and JSON output, tracking loss for each part separately.
    """

    def __init__(self, cot_weight=1.2, json_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cot_weight = cot_weight
        self.json_weight = json_weight
        self._cot_loss_sum = 0.0
        self._json_loss_sum = 0.0
        self._loss_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        token_weights = inputs.pop("token_weights", None)
        token_types = inputs.pop("token_types", None)

        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.shape)

        mask = (shift_labels != -100).float()

        if token_weights is not None:
            shift_weights = token_weights[..., 1:].contiguous()
            weighted_loss = (loss_per_token * shift_weights * mask).sum() / ((shift_weights * mask).sum() + 1e-8)

            if token_types is not None:
                shift_types = token_types[..., 1:].contiguous()
                cot_mask = ((shift_types == 1) & (mask == 1)).float()
                json_mask = ((shift_types == 2) & (mask == 1)).float()
            else:
                cot_mask = ((shift_weights == self.cot_weight) & (mask == 1)).float()
                json_mask = ((shift_weights == self.json_weight) & (mask == 1)).float()

            cot_token_count = cot_mask.sum().item()
            json_token_count = json_mask.sum().item()

            if cot_token_count > 0:
                cot_loss = (loss_per_token * cot_mask).sum() / cot_token_count
                self._cot_loss_sum += cot_loss.item()

            if json_token_count > 0:
                json_loss = (loss_per_token * json_mask).sum() / json_token_count
                self._json_loss_sum += json_loss.item()

            self._loss_count += 1
        else:
            weighted_loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)

        return (weighted_loss, outputs) if return_outputs else weighted_loss

    def log(self, logs, start_time=None):
        if self._loss_count > 0:
            logs["cot_loss"] = round(self._cot_loss_sum / self._loss_count, 4)
            logs["json_loss"] = round(self._json_loss_sum / self._loss_count, 4)
            self._cot_loss_sum = 0.0
            self._json_loss_sum = 0.0
            self._loss_count = 0

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


@dataclass
class WeightedDataCollator:
    """
    Custom DataCollator that preserves token_weights and token_types fields on top of standard padding.
    """
    tokenizer: Any
    padding: bool = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        has_token_weights = 'token_weights' in features[0]
        has_token_types = 'token_types' in features[0]

        token_weights_list = [f.pop('token_weights') for f in features] if has_token_weights else None
        token_types_list = [f.pop('token_types') for f in features] if has_token_types else None

        batch = self.tokenizer.pad(
            features, padding=self.padding, max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors,
        )

        if has_token_weights and token_weights_list:
            max_len = batch['input_ids'].shape[1]
            padded_weights = []
            for weights in token_weights_list:
                weights_tensor = weights if isinstance(weights, torch.Tensor) else torch.tensor(weights, dtype=torch.float32)
                weights_tensor = weights_tensor.to(torch.float32)
                current_len = len(weights_tensor)
                if current_len < max_len:
                    padded = torch.cat([weights_tensor, torch.zeros(max_len - current_len, dtype=torch.float32)])
                else:
                    padded = weights_tensor[:max_len]
                padded_weights.append(padded)
            batch['token_weights'] = torch.stack(padded_weights)

        if has_token_types and token_types_list:
            max_len = batch['input_ids'].shape[1]
            padded_types = []
            for types in token_types_list:
                types_tensor = types if isinstance(types, torch.Tensor) else torch.tensor(types, dtype=torch.long)
                types_tensor = types_tensor.to(torch.long)
                current_len = len(types_tensor)
                if current_len < max_len:
                    padded = torch.cat([types_tensor, torch.zeros(max_len - current_len, dtype=torch.long)])
                else:
                    padded = types_tensor[:max_len]
                padded_types.append(padded)
            batch['token_types'] = torch.stack(padded_types)

        return batch


class SyncEpochCallback(TrainerCallback):
    """Sync progressive distillation progress to collater at the beginning of each epoch."""

    def __init__(self, collater):
        self.collater = collater

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1
        self.collater.set_epoch(current_epoch)


def finetune(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("RANK") or 0)
    device = torch.device(local_rank)

    if ddp:
        device_map = {"": local_rank}
    else:
        device_map = "auto"

    if local_rank == 0:
        print(vars(args))

    # ---- 1. Load model and tokenizer ----
    checkpoint_path = args.resume_from_checkpoint
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        truncation_side='left',
        trust_remote_code=True,
        use_fast=True
    )

    extended_tokens = json.load(open(args.extended_tokens, 'r'))
    add_num = tokenizer.add_tokens(extended_tokens)
    config = AutoConfig.from_pretrained(checkpoint_path)
    config.vocab_size = len(tokenizer)
    print(f'{add_num} tokens added, total vocab size: {config.vocab_size}')

    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f'Model loaded in {round(time.time() - t0, 2)}s')
    model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()

    # ---- 2. Data loading ----
    cot_weight = args.cot_weight
    json_weight = args.json_weight
    distill_direction = args.distill_direction

    collater = ProgressiveCotDistillCollater(
        applied_tokenizer=True,
        tokenizer=tokenizer,
        max_length=args.model_max_length,
        cot_weight=cot_weight,
        json_weight=json_weight,
        distill_direction=distill_direction
    )

    train_data = CSVDataset(csv_path=args.train_csv, collate_fn=collater)
    print(f"Training samples: {len(train_data)}")

    data_collator = WeightedDataCollator(tokenizer=tokenizer, padding=True)

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ---- 3. Training configuration ----
    training_args = transformers.TrainingArguments(
        seed=args.seed,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        logging_steps=args.logging_step,
        optim=args.optim,
        gradient_checkpointing=True,
        eval_strategy="no",
        save_strategy="no",
        output_dir=args.output_dir,
        save_total_limit=1,
        load_best_model_at_end=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="none",
        dataloader_drop_last=True,
        ignore_data_skip=True,
        remove_unused_columns=False
    )

    epoch_sync_cb = SyncEpochCallback(collater)

    print(f"Using WeightedLossTrainer with cot_weight={cot_weight}, json_weight={json_weight}")
    trainer = WeightedLossTrainer(
        cot_weight=cot_weight,
        json_weight=json_weight,
        model=model,
        train_dataset=train_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[epoch_sync_cb]
    )

    model.config.use_cache = False
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=args.output_dir)

    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Progressive Implicit CoT Distillation Finetune')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    args = parser.parse_args()
    finetune(args)
