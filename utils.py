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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--logging_step", type=int, default=10)
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--cot_mode", type=str, default='progressive_cot_distill',
                        help="CoT mode (default: progressive_cot_distill)")
    parser.add_argument("--distill_direction", type=str, default='backward',
                        choices=['forward', 'backward'],
                        help="Distill direction: 'forward' (CONTEXT first) or 'backward' (last STEP first)")
    parser.add_argument("--cot_weight", type=float, default=0.5,
                        help="Loss weight for CoT tokens (default: 0.5)")
    parser.add_argument("--json_weight", type=float, default=1.0,
                        help="Loss weight for JSON tokens (default: 1.0)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to pretrained model or checkpoint")
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--deepspeed", type=str, default="./config/ds_z3_bf16.json")
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
