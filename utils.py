import os
import re
import random

import numpy as np
import torch


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./ckpt/", help="The output directory")
    parser.add_argument("--extended_tokens", type=str,
                        default="./add_tokens/extended_cot_vocabs.json",
                        help="The extended tokens path")
    return parser


def parse_dataset_args(parser):
    parser.add_argument("--train_csv", type=str, default=None, help="Path to training CSV file")
    parser.add_argument("--valid_csv", type=str, default=None, help="Path to validation/test CSV file")
    return parser


def parse_train_args(parser):
    parser.add_argument("--optim", type=str, default="adamw_torch", help='Optimizer name')
    parser.add_argument("--epochs", type=int, default=13)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--cot_mode", type=str, default='latent_multi_cot',
                        choices=['latent_multi_cot'],
                        help="PICD training target format.")
    parser.add_argument("--cot_weight", type=float, default=1.0,
                        help="Section weight for CoT/latent-prefix loss")
    parser.add_argument("--json_weight", type=float, default=1.0,
                        help="Section weight for JSON-plan loss")
    parser.add_argument("--picd_lr_schedule", type=str, default="compression_aware",
                        choices=["none", "compression_aware"],
                        help="PICD-aware LR schedule. Enabled by default for latent_multi_cot.")
    parser.add_argument("--picd_lr_target_compressed_blocks", type=int, default=9,
                        help="Keep structure LR until this many PICD semantic blocks have been compressed.")
    parser.add_argument("--picd_polish_lr", type=float, default=1e-6,
                        help="Low LR after the PICD structure phase.")
    parser.add_argument("--picd_polish_decay_epochs", type=float, default=0.0,
                        help="Optional polish cosine decay horizon in physical epochs.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to pretrained model or checkpoint")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
    parser.add_argument("--save_strategy", type=str, default="no", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=1)
    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def ensure_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)


def get_last_checkpoint(folder):
    PREFIX_CHECKPOINT_DIR = "checkpoint"
    _re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")
    content = os.listdir(folder)
    checkpoints = [
        path for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))
