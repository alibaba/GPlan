"""
GPlan Open Test Script

Adapted from g_plan branch's test_wsc.py for the open-source version.
Key differences from g_plan:
- Uses local CSV files (CSVDataset) instead of ODPS tables
- Uses ProgressiveCotDistillCollater instead of gplan_amap_odps_raw_collater
- Tool names are anonymized (tool_1 ~ tool_10)
- Metrics: First Intent Accuracy, Avg Weighted Edit Similarity, NDCG@3 (tool_tag)
- Supports DDP multi-GPU inference with metric aggregation
"""

import os
import argparse
import time
import random
import math
import json
import re

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import Dataset

from data_process.collate_fns import ProgressiveCotDistillCollater
from data_process.data_loader import CSVDataset
from utils import parse_global_args, parse_train_args, parse_dataset_args, set_seed, ensure_dir


# ============================================================
# Intent Extraction Utilities
# ============================================================

def extract_first_intent(text):
    """
    Extract the first intent dict from model output or label text.
    Supports: <THOUGHT>...</THOUGHT>[{...}, ...] or plain JSON arrays.
    """
    if not text or not isinstance(text, str):
        return None

    code_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    text = code_block.group(1) if code_block else text
    text = text.strip().lstrip('[]').strip()

    if not text:
        return None

    # Try to parse as JSON list
    start_idx = text.find('[')
    if start_idx != -1:
        stack = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '[':
                stack += 1
            elif text[i] == ']':
                stack -= 1
                if stack == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            json_candidate = text[start_idx: end_idx + 1]
            try:
                plan_list = json.loads(json_candidate)
                if isinstance(plan_list, list) and len(plan_list) > 0:
                    return plan_list[0]
            except Exception:
                pass

    # Fallback: extract first {} object
    try:
        obj_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        if obj_match:
            obj_str = obj_match.group(1).replace("'", '"')
            return json.loads(obj_str)
    except Exception:
        return None

    return None


def extract_intent_sequence(text):
    """
    Extract the full intent sequence from model output or label text.
    Handles truncated JSON (when output is cut by max_new_tokens).
    Returns: List[Dict]
    """
    if not text or not isinstance(text, str):
        return []

    code_block = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    text = code_block.group(1) if code_block else text
    text = text.strip()

    if not text:
        return []

    start_idx = text.find('[')
    if start_idx != -1:
        stack = 0
        end_idx = -1
        for i in range(start_idx, len(text)):
            if text[i] == '[':
                stack += 1
            elif text[i] == ']':
                stack -= 1
                if stack == 0:
                    end_idx = i
                    break

        if end_idx != -1:
            json_candidate = text[start_idx: end_idx + 1]
            try:
                plan_list = json.loads(json_candidate)
                if isinstance(plan_list, list):
                    return [item for item in plan_list if isinstance(item, dict)]
            except Exception:
                pass
        else:
            return _extract_all_complete_objects(text[start_idx:])

    return _extract_all_complete_objects(text)


def _extract_all_complete_objects(text):
    """Extract all complete JSON objects {} from text (handles truncated arrays)."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            stack = 0
            start = i
            end = -1
            for j in range(i, len(text)):
                if text[j] == '{':
                    stack += 1
                elif text[j] == '}':
                    stack -= 1
                    if stack == 0:
                        end = j
                        break

            if end != -1:
                obj_str = text[start:end + 1]
                try:
                    obj = json.loads(obj_str)
                    if isinstance(obj, dict) and '工具名称' in obj:
                        results.append(obj)
                except Exception:
                    pass
                i = end + 1
            else:
                break
        else:
            i += 1
    return results


# ============================================================
# Evaluation Metrics
# ============================================================

def intent_match_tool_tag(pred_intent, label_intent):
    """Match by tool name + tag parameter."""
    if pred_intent is None or label_intent is None:
        return False
    tool_match = pred_intent.get('工具名称') == label_intent.get('工具名称')
    tag_match = pred_intent.get('tag') == label_intent.get('tag')
    return tool_match and tag_match


def intent_substitution_cost(intent_a, intent_b, tool_mismatch_cost=1.0, param_mismatch_cost=0.3):
    """Compute substitution cost between two intents for weighted edit distance."""
    if intent_a == intent_b:
        return 0.0
    if intent_a.get('工具名称', '') != intent_b.get('工具名称', ''):
        return tool_mismatch_cost
    return param_mismatch_cost


def weighted_edit_distance(sequence_a, sequence_b, tool_mismatch_cost=1.0, param_mismatch_cost=0.3,
                           insertion_cost=1.0, deletion_cost=1.0):
    """Compute weighted edit distance between two intent sequences."""
    length_a, length_b = len(sequence_a), len(sequence_b)
    dp = [[0.0] * (length_b + 1) for _ in range(length_a + 1)]

    for i in range(length_a + 1):
        dp[i][0] = i * deletion_cost
    for j in range(length_b + 1):
        dp[0][j] = j * insertion_cost

    for i in range(1, length_a + 1):
        for j in range(1, length_b + 1):
            delete = dp[i - 1][j] + deletion_cost
            insert = dp[i][j - 1] + insertion_cost
            substitute = dp[i - 1][j - 1] + intent_substitution_cost(
                sequence_a[i - 1], sequence_b[j - 1], tool_mismatch_cost, param_mismatch_cost
            )
            dp[i][j] = min(delete, insert, substitute)

    return dp[length_a][length_b]


def normalized_edit_similarity(sequence_a, sequence_b, tool_mismatch_cost=1.0, param_mismatch_cost=0.3,
                               insertion_cost=1.0, deletion_cost=1.0):
    """Compute normalized edit distance similarity (0-1, 1 = identical)."""
    if not sequence_a and not sequence_b:
        return 1.0

    distance = weighted_edit_distance(sequence_a, sequence_b, tool_mismatch_cost, param_mismatch_cost,
                                      insertion_cost, deletion_cost)
    max_length = max(len(sequence_a), len(sequence_b))
    max_possible_distance = max_length * max(insertion_cost, deletion_cost, tool_mismatch_cost)

    if max_possible_distance == 0:
        return 1.0

    similarity = 1.0 - (distance / max_possible_distance)
    return max(0.0, min(1.0, similarity))


def ndcg_at_k_tool_tag(pred_seq, label_seq, k=3):
    """
    Compute NDCG@K using tool_tag matching (tool name + tag parameter).
    Relevance is 1 if a predicted intent matches any label intent by tool_tag, else 0.
    """
    if not label_seq:
        return 0.0

    relevance = []
    for pred_intent in pred_seq[:k]:
        rel = 0
        for label_intent in label_seq:
            if intent_match_tool_tag(pred_intent, label_intent):
                rel = 1
                break
        relevance.append(rel)

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance))

    ideal_length = min(k, len(label_seq))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_length))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# Main Test Function
# ============================================================

def test(args):
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    global_rank = int(os.environ.get("RANK") or 0)

    if global_rank == 0:
        print(vars(args))

    device = torch.device(f"cuda:{local_rank}")
    print(f'world_size: {world_size}, local_rank: {local_rank}, global_rank: {global_rank}, device: {device}')

    if ddp:
        device_map = {"": local_rank}
        dist.init_process_group(backend="nccl")
    else:
        device_map = "auto"

    # ---- 1. Load model and tokenizer ----
    checkpoint_path = args.resume_from_checkpoint
    if not checkpoint_path or not os.path.isdir(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    print(f'Loading tokenizer from {checkpoint_path}...')
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        truncation_side='left',
        trust_remote_code=True,
        use_fast=True
    )

    print(f'Loading model from {checkpoint_path}...')
    load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, device_map=device_map, trust_remote_code=True
    )
    print(f'Model loaded in {round(time.time() - load_start, 2)}s')

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # ---- 2. Load validation data from CSV ----
    collater = ProgressiveCotDistillCollater(
        applied_tokenizer=False,
        tokenizer=tokenizer
    )

    valid_csv_path = args.valid_csv
    if not valid_csv_path:
        raise ValueError("--valid_csv is required for testing")

    valid_data = CSVDataset(csv_path=valid_csv_path, collate_fn=collater)
    print(f"[Rank {global_rank}] Validation samples: {len(valid_data)}")

    # DDP: shard data across ranks
    if ddp:
        all_indices = list(range(len(valid_data)))
        shard_indices = all_indices[global_rank::world_size]
        valid_data_list = [valid_data[i] for i in shard_indices]
    else:
        valid_data_list = [valid_data[i] for i in range(len(valid_data))]

    valid_dataset_text = Dataset.from_list(valid_data_list)

    if valid_dataset_text and global_rank == 0:
        print(f"\n=============== Checkpoint: Raw text sample ===============")
        sample_idx = random.randint(0, len(valid_dataset_text) - 1)
        print(f"Sample {sample_idx}: {valid_dataset_text[sample_idx]}")
        print("============================================================\n")

    # ---- 3. Tokenize for inference ----
    def tokenize_for_inference(batch_examples):
        input_ids_list = []
        attention_mask_list = []
        label_text_list = []

        for msgs in batch_examples['messages']:
            prompt_messages = msgs[:-1]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            prompt_enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=args.model_max_length,
                padding=False,
                add_special_tokens=False,
                return_tensors="pt"
            )

            label_content = msgs[-1]['content']

            input_ids_list.append(prompt_enc.input_ids[0])
            attention_mask_list.append(prompt_enc.attention_mask[0])
            label_text_list.append(label_content)

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "label_text": label_text_list
        }

    tokenized_dataset = valid_dataset_text.map(
        tokenize_for_inference,
        batched=True,
        remove_columns=valid_dataset_text.column_names,
        desc="Preparing inference data"
    )

    if tokenized_dataset and global_rank == 0:
        print(f"\n============== Checkpoint: Tokenized sample ==============")
        sample_idx = random.randint(0, len(tokenized_dataset) - 1)
        tokenized_sample = tokenized_dataset[sample_idx]
        sample_input_ids = tokenized_sample['input_ids']
        print(f"Sample {sample_idx}:")
        print(f"  Input IDs (first 20): {sample_input_ids[:20]}...")
        print(f"  Sequence length: {len(sample_input_ids)}")
        decoded_prompt = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
        print(f"  Decoded prompt (partial):\n'{decoded_prompt.replace(tokenizer.pad_token, '')[:500]}...'")
        print("============================================================\n")

    # ---- 4. Batch generation ----
    has_printed_generate_input = False

    @torch.inference_mode()
    def batch_generate(model, tokenizer, dataset, batch_size=8, device=None):
        nonlocal has_printed_generate_input
        results = []
        target_device = device if device is not None else torch.device("cuda:0")

        for i in tqdm(range(0, len(dataset), batch_size),
                      desc=f"[Rank {global_rank}] Generating",
                      disable=(global_rank != 0)):
            batch = dataset[i:i + batch_size]

            raw_input_ids = [torch.tensor(x) for x in batch["input_ids"]]
            padded_inputs = tokenizer.pad(
                {"input_ids": raw_input_ids},
                padding=True,
                return_tensors="pt",
            )
            input_ids = padded_inputs["input_ids"].to(target_device)
            attention_mask = padded_inputs["attention_mask"].to(target_device)
            batch_label_texts = batch["label_text"]
            indices = list(range(i, min(i + batch_size, len(dataset))))

            if not has_printed_generate_input and global_rank == 0:
                print("\n" + "=" * 20 + " Model Input Sanity Check (printed once) " + "=" * 20)
                sample_in_batch = random.randint(0, len(input_ids) - 1)
                single_mask = attention_mask[sample_in_batch]
                last_real = (single_mask == 1).nonzero(as_tuple=True)[0].max()
                effective_ids = input_ids[sample_in_batch][:last_real + 1]
                decoded_input = tokenizer.decode(effective_ids, skip_special_tokens=False)
                print(f"\nDecoded model input:\n{decoded_input}")
                print(f"\nGround truth label:\n'{batch_label_texts[sample_in_batch]}'")
                has_printed_generate_input = True

            input_ids_length = input_ids.shape[1]
            im_end_token_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)[0]

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.model_max_length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_token_id
            )

            newly_generated_ids = outputs[:, input_ids_length:]

            for idx, pred_ids, label_text in zip(indices, newly_generated_ids.tolist(), batch_label_texts):
                cleaned_pred_ids = [
                    token_id for token_id in pred_ids
                    if token_id != tokenizer.pad_token_id and token_id != tokenizer.eos_token_id
                ]
                results.append({
                    'index': idx,
                    'pred_ids': cleaned_pred_ids,
                    'label_text': label_text
                })

        return results

    # ---- 5. Run inference ----
    results = batch_generate(model, tokenizer, tokenized_dataset,
                             batch_size=args.test_batch_size, device=device)

    # ---- 6. Compute metrics ----
    first_intent_correct = 0
    total_edit_similarity = 0.0
    total_ndcg_at_3 = 0.0
    total = 0

    # Random samples to print (rank 0 only)
    samples_to_print = []
    if global_rank == 0:
        num_print = min(5, len(results))
        samples_to_print = sorted(random.sample(range(len(results)), num_print))

    for idx, res in enumerate(results):
        pred_text = tokenizer.decode(res['pred_ids'], skip_special_tokens=True)
        label_text = res['label_text']

        pred_intent = extract_first_intent(pred_text)
        label_intent = extract_first_intent(label_text)

        pred_seq = extract_intent_sequence(pred_text)
        label_seq = extract_intent_sequence(label_text)

        # First intent accuracy (exact match)
        if pred_intent and label_intent and pred_intent == label_intent:
            first_intent_correct += 1

        # Weighted edit distance similarity
        edit_sim = normalized_edit_similarity(
            pred_seq, label_seq,
            tool_mismatch_cost=args.tool_mismatch_cost,
            param_mismatch_cost=args.param_mismatch_cost
        )
        total_edit_similarity += edit_sim

        # NDCG@3 (tool_tag mode)
        ndcg_val = ndcg_at_k_tool_tag(pred_seq, label_seq, k=3)
        total_ndcg_at_3 += ndcg_val

        total += 1

        # Print random samples on rank 0
        if global_rank == 0 and idx in samples_to_print:
            print(f"\n--- Sample {idx} ---")
            print(f"  Model output: {pred_text}")
            if pred_intent is None:
                print(f"  [Parse failed] Raw: {pred_text[:150]}...")
            else:
                print(f"  Pred first intent: {pred_intent}")
            print(f"  Label first intent: {label_intent}")
            print(f"  Pred sequence ({len(pred_seq)}): {pred_seq}")
            print(f"  Label sequence ({len(label_seq)}): {label_seq}")
            print(f"  First intent match: {pred_intent == label_intent}")
            print(f"  Edit similarity: {edit_sim:.4f}")
            print(f"  NDCG@3 (tool_tag): {ndcg_val:.4f}")

    # ---- 7. Aggregate across ranks (DDP) ----
    if ddp:
        local_correct = torch.tensor([first_intent_correct], dtype=torch.long, device=device)
        local_total = torch.tensor([total], dtype=torch.long, device=device)
        local_edit_sim = torch.tensor([total_edit_similarity], dtype=torch.float64, device=device)
        local_ndcg = torch.tensor([total_ndcg_at_3], dtype=torch.float64, device=device)

        dist.all_reduce(local_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_edit_sim, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_ndcg, op=dist.ReduceOp.SUM)

        first_intent_correct = local_correct.item()
        total = local_total.item()
        total_edit_similarity = local_edit_sim.item()
        total_ndcg_at_3 = local_ndcg.item()

    # ---- 8. Print final results (rank 0 only) ----
    if global_rank == 0:
        accuracy = first_intent_correct / total if total > 0 else 0
        avg_edit_sim = total_edit_similarity / total if total > 0 else 0
        avg_ndcg_at_3 = total_ndcg_at_3 / total if total > 0 else 0

        print(f"\n{'=' * 60}")
        print(f"  Final Metrics ({total} samples)")
        print(f"{'=' * 60}")
        print(f"  First Intent Accuracy:        {accuracy:.4f} ({first_intent_correct}/{total})")
        print(f"  Avg Weighted Edit Similarity: {avg_edit_sim:.4f}")
        print(f"  NDCG@3 (tool_tag):            {avg_ndcg_at_3:.4f}")
        print(f"{'=' * 60}")
        print(f"\n  [Config]")
        print(f"  tool_mismatch_cost: {args.tool_mismatch_cost}")
        print(f"  param_mismatch_cost: {args.param_mismatch_cost}")
        print(f"{'=' * 60}")

    # Cleanup
    if ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPlan Open Test')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    parser.add_argument("--test_batch_size", type=int, default=4,
                        help="Batch size for inference")
    parser.add_argument("--tool_mismatch_cost", type=float, default=1.0,
                        help="Substitution cost when tool names differ")
    parser.add_argument("--param_mismatch_cost", type=float, default=0.3,
                        help="Substitution cost when tool matches but params differ")

    args = parser.parse_args()
    test(args)
