# GPlan: Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap

<div align="center">

AMAP, Alibaba Group

</div>

## 📋 Overview

Progressive Implicit CoT Distillation (PICD) training framework of GPlan. It uses **curriculum learning** to compress structured CoT text into fixed-length latent think-token blocks and uses a compression-aware learning-rate schedule (CALR) for the structure-to-polish transition.

<div align="center">
<img src="gplan_framework.png" alt="GPlan Framework" width="100%"/>
</div>

## 📊GSISR Dataset

The GSISR dataset is collected from Amap and provided in `data_process/dataset/`. All data have been **anonymized** to protect privacy — original feature names, POI identifiers, and user identifiers have been replaced with generic placeholders.

| File | Description | Num     |
|:-----|:------------|:--------|
| `data_process/dataset/train.csv` | Training set | 100,000 |
| `data_process/dataset/test.csv` | Test set | 1,000   |

### User Profiles and Behavior History

Each user is described by 14 anonymized profile features. All categorical values have been mapped to numerical IDs.

| Field                | Description                                                                                                                        |
|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| User ID              | A unique numerical identifier for each user.                                                                                       |
| Profile Feature 1–14 | Anonymized user profile attributes.                                                                                                |
| Short-term Behavior Seq | Anonymized short-term behavior sequence. POI names and behavior types (e.g., click) are replaced with numerical IDs (`p_`, `act_`). |
| Long-term Behavior | Anonymized long-term behavior feature. Original values are replaced with numerical IDs.                                            |

### Context Information

| Field | Description                                                            |
|:---|:-----------------------------------------------------------------------|
| Current Time | Time of the request.                                                   |
| Weekend Flag | Whether the current day is a weekend (0/1).                            |
| Holiday Flag | Whether the current day is a holiday (0/1).                            |
| Current City & District | The city and district where the user is located.                       |
| Current POI Name | The name of the user's current Point of Interest (mapped to ID, `p_`). |
| Current POI Category | The tag of the current POI.                                            |

### Trigger Events

Each request includes 7 trigger event features that capture the user's immediate intent signals.

| Field | Description |
|:---|:---|
| Trigger 1–7 | Anonymized event trigger features. Original event types and descriptions are replaced with numerical IDs or kept as timestamps. |


### Labels

Each label is an **intent sequence** — a JSON array of tool-calling intents representing the recommendation. Each intent includes a tool name and associated parameters selected from a predefined intent library:

```json
[
  {"工具名称": "tool_5", "起始位置": "当前位置", "空间范围": "附近", "tag": "美食"},
  {"工具名称": "tool_2", "起始位置": "当前位置", "终点位置": "家"},
  ...
]
```

The intent library includes 10 tool types covering scenarios such as ride-hailing, navigation, transit, POI recommendation, order reminders, weather queries, etc.

### PICD Training Data Preparation

The released CSV files provide final intent sequences as labels. To run PICD training, prepare a CoT-augmented CSV with the same schema as `train.csv`, where `raw_labels` contains a structured CoT followed by the final intent sequence.

**Step 1: Prepare structured CoT**

For each training sample, generate a concise reasoning trace from the user profile, behavior history, current context, and gold intent sequence. The CoT should explain why the plan is reasonable, with each `<STEP_n>` aligned to the n-th intent in the JSON label:

```
<THOUGHT>
<CONTEXT>Briefly analyze the current context and user profile</CONTEXT>
<STRATEGY>Describe the planning strategy</STRATEGY>
<STEP_1>Explain the first recommended intent</STEP_1>
...
<STEP_n>Explain the n-th recommended intent</STEP_n>
</THOUGHT>
```

The number of `<STEP_n>` fields should match the number of intents in the JSON array.

**Step 2: Write `raw_labels`**

Concatenate the CoT and JSON intent sequence in `raw_labels`:

```
<THOUGHT><CONTEXT>...</CONTEXT><STRATEGY>...</STRATEGY><STEP_1>...</STEP_1><STEP_2>...</STEP_2><STEP_3>...</STEP_3></THOUGHT>[{"工具名称":"tool_5","起始位置":"当前位置","空间范围":"附近","tag":"美食"},{"工具名称":"tool_2","起始位置":"当前位置","终点位置":"家"},{"工具名称":"tool_7","tag":"景点"}]
```

The collator parses this field and applies progressive implicit CoT distillation automatically.

## 🚀 Quick Start

### Evaluation on the Public Dataset

```bash
pip install -r requirements.txt
bash test.sh
```

The test script reports the offline metrics used in the paper: `Acc@1`, `NDCG@3`, and `NES` (normalized edit similarity). It can use the JSON-only labels included in `data_process/dataset/test.csv`.

### PICD Training

```bash
TRAIN_CSV=/path/to/cot_augmented_train.csv bash finetune.sh
```

`finetune.sh` uses `--cot_mode=latent_multi_cot` and expects CoT-augmented `raw_labels`.

## 📁 Project Structure

```
├── finetune.py                         # Training script (WeightedLossTrainer + SyncEpochCallback)
├── finetune.sh                         # Training launch script
├── test.py                             # Test script (Acc@1, NDCG@3, NES)
├── test.sh                             # Test launch script
├── data_process/
│   ├── dataset/
│   │   ├── train.csv                   # Training dataset (anonymized)
│   │   └── test.csv                    # Test dataset (anonymized)
│   ├── collate_fns.py                  # PICD data collator
│   └── data_loader.py                  # CSV data loading
├── utils.py                            # Utility functions and argument definitions
├── add_tokens/extended_cot_vocabs.json  # CoT special token vocabulary
├── config/ds_z3_bf16.json              # DeepSpeed ZeRO-3 configuration
└── requirements.txt
```
