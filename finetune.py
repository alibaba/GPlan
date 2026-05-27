"""
GPlan PICD finetuning script.

Key features:
1. PICD with K=3 latent tokens per reasoning block.
2. Section-normalized CoT/JSON loss.
3. Compression-aware LR (CALR).
"""

import os
import json
import csv
import argparse
import time
import math

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

from data_process.collate_fns import LATENT_MODES, ProgressiveCotDistillCollater, normalize_cot_mode
from data_process.data_loader import CSVDataset
from utils import parse_global_args, parse_train_args, parse_dataset_args, set_seed, ensure_dir


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def load_extended_tokens(path, cot_mode):
    with open(path, "r", encoding="utf-8") as handle:
        token_config = json.load(handle)

    mode = normalize_cot_mode(cot_mode)
    if isinstance(token_config, dict):
        if mode not in token_config:
            raise ValueError(f"{path} does not contain token config for cot_mode={mode}")
        tokens = token_config[mode]
    else:
        tokens = token_config

    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError(f"{path} must provide a list of string tokens")
    return tokens


def _looks_like_picd_label(raw_label):
    text = str(raw_label or "")
    return "<THOUGHT>" in text and "</THOUGHT>" in text and "[" in text


def validate_picd_training_csv(csv_path):
    if not csv_path:
        raise ValueError("--train_csv is required for PICD training")
    if not os.path.isfile(csv_path):
        raise ValueError(f"Training CSV not found: {csv_path}")

    row_count = 0
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "raw_labels" not in reader.fieldnames:
            raise ValueError("Training CSV must contain a raw_labels column")
        for row in reader:
            row_count += 1
            if _looks_like_picd_label(row.get("raw_labels")):
                return

    raise ValueError(
        f"{csv_path} contains {row_count} rows but no PICD CoT labels. "
        "The public GSISR CSVs contain final JSON labels only; the CoT text used for PICD training "
        "is not included in this release. Provide a private or externally generated CSV whose "
        "raw_labels values contain <THOUGHT>...</THOUGHT> followed by the JSON plan."
    )


class CompressionAwareLRSchedulerMixin:
    """Trainer mixin for PICD's structure-then-polish LR schedule."""

    def __init__(self, *args, picd_lr_config=None, **kwargs):
        self.picd_lr_config = picd_lr_config or {}
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        cfg = getattr(self, "picd_lr_config", None) or {}
        if cfg.get("schedule") != "compression_aware":
            return super().create_scheduler(num_training_steps, optimizer)

        if self.lr_scheduler is not None:
            return self.lr_scheduler

        optimizer = optimizer if optimizer is not None else self.optimizer
        if optimizer is None:
            return super().create_scheduler(num_training_steps, optimizer)

        structure_lr = max(float(cfg.get("structure_lr", self.args.learning_rate)), 1e-12)
        polish_lr = float(cfg.get("polish_lr", 1e-6))
        polish_ratio = max(polish_lr / structure_lr, 0.0)
        target_blocks = max(0, int(cfg.get("target_compressed_blocks", 9)))
        polish_decay_epochs = max(0.0, float(cfg.get("polish_decay_epochs", 0.0) or 0.0))

        structure_epochs = target_blocks + 1
        epochs = max(float(getattr(self.args, "num_train_epochs", 1.0) or 1.0), 1e-8)
        steps_per_epoch = max(float(num_training_steps) / epochs, 1.0)
        transition_step = int(round(structure_epochs * steps_per_epoch))
        transition_step = min(max(transition_step, 0), num_training_steps)
        warmup_steps = int(self.args.get_warmup_steps(num_training_steps))
        remaining_polish_steps = max(0, num_training_steps - transition_step)
        if polish_decay_epochs > 0:
            polish_steps = max(1, int(round(polish_decay_epochs * steps_per_epoch)))
        else:
            polish_steps = max(1, remaining_polish_steps)

        if os.environ.get("RANK", "0") == "0":
            print(
                "PICD compression-aware LR enabled: "
                f"structure_lr={structure_lr}, polish_lr={polish_lr}, "
                f"target_compressed_blocks={target_blocks}, "
                f"structure_epochs={structure_epochs}, transition_step={transition_step}, "
                f"polish_steps={polish_steps}, num_training_steps={num_training_steps}, "
                f"warmup_steps={warmup_steps}"
            )

        def lr_lambda(current_step: int):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < transition_step or transition_step >= num_training_steps:
                return 1.0
            progress = min(1.0, max(0.0, float(current_step - transition_step) / polish_steps))
            return polish_ratio * 0.5 * (1.0 + math.cos(math.pi * progress))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return self.lr_scheduler


class WeightedLossTrainer(CompressionAwareLRSchedulerMixin, transformers.Trainer):
    """
    Custom Trainer with token-level loss weighting.
    Applies different weights to CoT text and JSON output, tracking loss for each part separately.
    """

    def __init__(self, cot_weight=1.0, json_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cot_weight = float(cot_weight)
        self.json_weight = float(json_weight)
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
            section_losses = []
            section_weights = []

            if cot_token_count > 0:
                cot_loss = (loss_per_token * cot_mask).sum() / cot_token_count
                self._cot_loss_sum += cot_loss.item()
                section_losses.append(self.cot_weight * cot_loss)
                section_weights.append(self.cot_weight)

            if json_token_count > 0:
                json_loss = (loss_per_token * json_mask).sum() / json_token_count
                self._json_loss_sum += json_loss.item()
                section_losses.append(self.json_weight * json_loss)
                section_weights.append(self.json_weight)

            if section_losses:
                weighted_loss = sum(section_losses) / (sum(section_weights) + 1e-8)
            else:
                weighted_loss = (loss_per_token * shift_weights * mask).sum() / ((shift_weights * mask).sum() + 1e-8)
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
        features = [dict(feature) for feature in features]
        has_labels = 'labels' in features[0]
        has_token_weights = 'token_weights' in features[0]
        has_token_types = 'token_types' in features[0]

        labels_list = [f.pop('labels') for f in features] if has_labels else None
        token_weights_list = [f.pop('token_weights') for f in features] if has_token_weights else None
        token_types_list = [f.pop('token_types') for f in features] if has_token_types else None

        batch = self.tokenizer.pad(
            features, padding=self.padding, max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors,
        )

        def pad_1d(values, max_len, pad_value, dtype):
            tensor = values if isinstance(values, torch.Tensor) else torch.tensor(values, dtype=dtype)
            tensor = tensor.to(dtype)
            current_len = len(tensor)
            if current_len < max_len:
                padding = torch.full((max_len - current_len,), pad_value, dtype=dtype)
                return torch.cat([tensor, padding])
            return tensor[:max_len]

        if has_labels and labels_list:
            max_len = batch['input_ids'].shape[1]
            batch['labels'] = torch.stack([
                pad_1d(labels, max_len, -100, torch.long)
                for labels in labels_list
            ])

        if has_token_weights and token_weights_list:
            max_len = batch['input_ids'].shape[1]
            batch['token_weights'] = torch.stack([
                pad_1d(weights, max_len, 0.0, torch.float32)
                for weights in token_weights_list
            ])

        if has_token_types and token_types_list:
            max_len = batch['input_ids'].shape[1]
            batch['token_types'] = torch.stack([
                pad_1d(types, max_len, 0, torch.long)
                for types in token_types_list
            ])

        return batch


class SyncEpochCallback(TrainerCallback):
    """Sync physical epochs to PICD compression stages."""

    def __init__(self, collater):
        self.collater = collater

    def on_epoch_begin(self, args, state, control, **kwargs):
        physical_epoch = int(state.epoch or 0) + 1
        self.collater.set_epoch(physical_epoch)
        if os.environ.get("RANK", "0") == "0":
            print(f"PICD epoch sync: physical_epoch={physical_epoch}")


def finetune(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    mode = normalize_cot_mode(args.cot_mode)
    validate_picd_training_csv(args.train_csv)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
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

    add_num = 0
    config = AutoConfig.from_pretrained(checkpoint_path)
    if args.extended_tokens and args.extended_tokens.lower() not in {"none", "null", ""} and mode in LATENT_MODES:
        extended_tokens = load_extended_tokens(args.extended_tokens, mode)
        add_num = tokenizer.add_tokens(extended_tokens)
        config.vocab_size = len(tokenizer)
    print(f'{add_num} tokens added, total vocab size: {config.vocab_size}')

    if local_rank == 0:
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f'Model loaded in {round(time.time() - t0, 2)}s')
    if add_num:
        model.resize_token_embeddings(len(tokenizer))
    model.enable_input_require_grads()

    # ---- 2. Data loading ----
    cot_weight = args.cot_weight
    json_weight = args.json_weight

    collater = ProgressiveCotDistillCollater(
        mode=mode,
        applied_tokenizer=True,
        tokenizer=tokenizer,
        max_length=args.model_max_length,
        cot_weight=cot_weight,
        json_weight=json_weight,
    )
    collater.set_epoch(1)

    train_data = CSVDataset(csv_path=args.train_csv, collate_fn=collater)
    print(f"Training samples: {len(train_data)}")
    if len(train_data) == 0:
        raise ValueError(
            "No valid PICD training samples. raw_labels must contain "
            "<THOUGHT>...</THOUGHT> followed by the JSON plan."
        )

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
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        output_dir=args.output_dir,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        deepspeed=args.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="none",
        dataloader_drop_last=True,
        ignore_data_skip=True,
        remove_unused_columns=False
    )

    epoch_sync_cb = SyncEpochCallback(collater)

    picd_lr_schedule = args.picd_lr_schedule
    if picd_lr_schedule != "none" and mode != "latent_multi_cot":
        if local_rank == 0:
            print(f"[WARNING] Disable picd_lr_schedule={picd_lr_schedule} for cot_mode={mode}.")
        picd_lr_schedule = "none"
    picd_lr_config = {
        "schedule": picd_lr_schedule,
        "structure_lr": args.learning_rate,
        "polish_lr": args.picd_polish_lr,
        "polish_decay_epochs": args.picd_polish_decay_epochs,
        "target_compressed_blocks": args.picd_lr_target_compressed_blocks,
    }

    print(
        "Using WeightedLossTrainer with "
        f"cot_weight={cot_weight}, json_weight={json_weight}, "
        f"picd_lr_schedule={picd_lr_schedule}"
    )
    trainer = WeightedLossTrainer(
        cot_weight=cot_weight,
        json_weight=json_weight,
        model=model,
        train_dataset=train_data,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[epoch_sync_cb],
        picd_lr_config=picd_lr_config
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
