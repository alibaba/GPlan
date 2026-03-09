"""
Progressive Implicit CoT Distillation Collater

Core idea: Through curriculum learning, progressively compress structured CoT semantic tags
(CONTEXT/STRATEGY/STEP_n) into special tokens (each tag -> 3 tokens) epoch by epoch,
distilling implicit reasoning capabilities.
"""

import json
import re
import codecs
import inspect
import torch


class ProgressiveCotDistillCollater:
    """
    During training, progressively compress structured CoT tags into special tokens via curriculum learning:
    - Epoch 1: Keep full CoT text
    - Epoch 2: Compress 1 tag into 3 special tokens
    - Epoch 3: Compress 2 tags
    - ...
    - Epoch N: All tags are compressed into special tokens

    Supports both 'forward' (front-to-back) and 'backward' (back-to-front) compression directions.
    """

    def __init__(self, tokenizer=None, max_length=None, applied_tokenizer=False,
                 cot_weight=1.0, json_weight=1.0, distill_direction='forward'):
        self.mode = 'progressive_cot_distill'
        self.epoch = 1
        self.distill_direction = distill_direction
        self.last_printed_epoch = 0
        self.applied_tokenizer = applied_tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if applied_tokenizer and max_length is None:
            raise ValueError("max_length is required when applied_tokenizer=True")
        self.max_length = max_length

        self.cot_weight = cot_weight
        self.json_weight = json_weight

        self.SYSTEM_PROMPT = """你是一个顶级的地图出行规划专家。你的任务是基于用户的综合信息，预测并生成一个符合逻辑、有时空感知的意图规划序列。
#核心策略
**重点分析用户当下最核心的第一个意图，后续意图序列是第一个意图的后续意图或者其他并列意图。**整个序列必须逻辑自洽、符合普通人的行为模式，并对时间和地点敏感（如饭点推美食），**序列有一定丰富度，长度大于3**
#输出规则
1.严格从意图库选择工具，根据公参填充参数，确保时空合理性，不得输出任何其它tag（包括但不限于：便利店、咖啡、饮品、公园、度假村、小吃快餐、夜宵、超市等）
2.你的输出必须由高度浓缩思维链和意图规划组成，思维链必须使用XML标签`<THOUGHT>...</THOUGHT>`包裹，意图规划必须使用JSON数组包裹，**XML中的<STEP_n>标签数量必须严格等于JSON数组中的意图数量**，模板如下: 
<THOUGHT>
<CONTEXT>简要分析当前时空以及用户整体画像</CONTEXT>
<STRATEGY>基于上下文分析，思考本次规划的核心策略</STRATEGY>
<STEP_1>重点分析当下最核心的第一个意图</STEP_1>
<STEP_2>说明为什么推荐第二个意图</STEP_2>
...
<STEP_n>说明为什么推荐第n个意图</STEP_n>
</THOUGHT>
[{"工具名称":"...","参数名":"..."}] //必须从意图库中选择工具名称，若该意图存在参数则从公参枚举中选择并填充，一共n个意图
#可用意图库
##公参枚举
-起始位置:["当前位置","前一位推荐结果","订单目的地"]
-终点位置:["当前位置","前一位推荐结果","订单目的地","家","公司","用户短期意向","调用猜你去哪模型"]
-空间范围:["附近","商圈","全城"]
-tag:["美食","购物","酒店","景点","休闲娱乐","运动健身","丽人","加油站","充电站","停车场"]
##意图
-{"工具名称":"tool_1","起始位置":"...","终点位置": "..."}//打车
-{"工具名称":"tool_2","起始位置":"...","终点位置": "..."}//出行
-{"工具名称":"tool_3"}//长途交通
-{"工具名称":"tool_4"}//周边公交
-{"工具名称":"tool_5","起始位置":"...","空间范围":"...","tag":"..."}//兴趣推荐
-{"工具名称":"tool_6"}//单poi真实到店展示
-{"工具名称":"tool_7","tag":"..."}//附近单poi推荐
-{"工具名称":"tool_8"}//订单提醒
-{"工具名称":"tool_9"}//查天气
-{"工具名称":"tool_10"}//写评价
#严格要求
-绝对禁止输出枚举外的值或工具名
-互斥组：{tool_1, tool_2, tool_4}在一个plan中最多出现一次
-终点位置/起始位置禁止输出具体POI名称，只能填枚举值
-若想表达的tag不在枚举内，必须就近映射（如：便利店/超市/零食/日杂→购物，咖啡/饮品/奶茶/甜品/夜宵/小吃快餐/餐厅/家常菜→美食，公园/度假村/自然风光/景区→景点，KTV/酒吧/Livehouse→休闲娱乐）"""

    def _parse_csv_row(self, csv_row):
        """Extract fields from a CSV dict row."""
        if not csv_row or not isinstance(csv_row, dict):
            return {}
        field_names = [
            'id',
            'profile_feat_1', 'profile_feat_2', 'profile_feat_3', 'profile_feat_4',
            'profile_feat_5', 'profile_feat_6', 'profile_feat_7',
            'profile_feat_8', 'profile_feat_9', 'profile_feat_10',
            'profile_feat_11', 'profile_feat_12', 'profile_feat_13', 'profile_feat_14',
            'short_term_behavior_seq', 'long_term_behavior',
            'currenttime',
            'holidayflag', 'weekendflag', 'current_poi_name', 'current_tag',
            'current_city', 'current_district',
            'trigger_1', 'trigger_2', 'trigger_3', 'trigger_4', 'trigger_5',
            'trigger_6', 'trigger_7',
            'raw_labels'
        ]
        data = {}
        for field in field_names:
            data[field] = csv_row.get(field, '')
        return data

    def set_epoch(self, epoch):
        """Set current epoch to control progressive distillation progress."""
        self.epoch = epoch

    def _progressive_distill_multi(self, thought_content):
        """
        Multi-token progressive distillation: replace each tag's content with 3 special placeholder tokens.
        Dynamically detects STEP tags present in the current sample to handle varying STEP counts (3-9).
        CONTEXT, STRATEGY, and each STEP are uniformly compressed into 3 tokens.
        """
        mapping = {
            "CONTEXT": "<THOUGHT_CONTEXT_a><THOUGHT_CONTEXT_b><THOUGHT_CONTEXT_c>",
            "STRATEGY": "<THOUGHT_STRATEGY_a><THOUGHT_STRATEGY_b><THOUGHT_STRATEGY_c>",
            "STEP_1": "<T_1_a><T_1_b><T_1_c>",
            "STEP_2": "<T_2_a><T_2_b><T_2_c>",
            "STEP_3": "<T_3_a><T_3_b><T_3_c>",
            "STEP_4": "<T_4_a><T_4_b><T_4_c>",
            "STEP_5": "<T_5_a><T_5_b><T_5_c>",
            "STEP_6": "<T_6_a><T_6_b><T_6_c>",
            "STEP_7": "<T_7_a><T_7_b><T_7_c>",
            "STEP_8": "<T_8_a><T_8_b><T_8_c>",
            "STEP_9": "<T_9_a><T_9_b><T_9_c>"
        }

        all_possible_steps = [
            "STEP_1", "STEP_2", "STEP_3", "STEP_4", "STEP_5",
            "STEP_6", "STEP_7", "STEP_8", "STEP_9"
        ]
        present_steps = [s for s in all_possible_steps if f"<{s}>" in thought_content]

        if self.distill_direction == 'forward':
            fold_order = ["CONTEXT", "STRATEGY"] + present_steps
        else:
            fold_order = list(reversed(present_steps)) + ["STRATEGY", "CONTEXT"]

        num_to_fold = max(0, int(self.epoch) - 1)
        tags_to_fold = fold_order[:num_to_fold]

        distilled_text = thought_content
        for tag in tags_to_fold:
            placeholder = mapping.get(tag, "")
            pattern = rf"<{tag}>.*?</{tag}>"
            distilled_text = re.sub(pattern, placeholder, distilled_text, flags=re.DOTALL)
        return distilled_text

    def _clean_assistant_response(self, raw_output):
        """
        Clean assistant response: extract THOUGHT and JSON, apply progressive distillation, combine output.
        Output format: <THOUGHT>...</THOUGHT>[{...}]
        """
        if not raw_output:
            return None

        full_content = raw_output

        thought_match = re.search(r'<THOUGHT>(.*?)</THOUGHT>', full_content, re.DOTALL)
        json_match = re.search(r'```json\s*(.*?)\s*```', full_content, re.DOTALL)

        thought_content = thought_match.group(1).strip() if thought_match else ""
        json_raw = json_match.group(1).strip() if json_match else ""

        if not json_raw:
            fallback_json = re.search(r'(\[.*\])', full_content, re.DOTALL)
            if fallback_json:
                json_raw = fallback_json.group(1).strip()

        try:
            plan_obj = json.loads(json_raw)
            clean_json_str = json.dumps(plan_obj, ensure_ascii=False, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            return None

        if thought_content:
            thought_distilled = self._progressive_distill_multi(thought_content)
            thought_compressed = re.sub(r'\s+', ' ', thought_distilled)
            thought_compressed = re.sub(r'>\s+<', '><', thought_compressed)
            clean_thought_tag = f"<THOUGHT>{thought_compressed.strip()}</THOUGHT>"
        else:
            clean_thought_tag = ""

        return f"{clean_thought_tag}{clean_json_str}" if clean_thought_tag else clean_json_str

    def _get_empty_sample(self):
        """Return None to indicate this sample should be filtered out."""
        return None

    def _find_cot_json_boundary(self, input_ids, assistant_response, tokenizer):
        """
        Locate the boundary between CoT and JSON in the token sequence.
        Format: <THOUGHT>...</THOUGHT>[{...}]

        Returns:
            tuple: (cot_start_pos, cot_end_pos, json_start_pos, json_end_pos)
        """
        cot_start_pos, cot_end_pos = -1, -1
        json_start_pos, json_end_pos = -1, -1

        thought_end_str = '</THOUGHT>'
        thought_end_tokens = tokenizer.encode(thought_end_str, add_special_tokens=False)

        for i in range(len(input_ids) - len(thought_end_tokens) + 1):
            match = True
            for j, token in enumerate(thought_end_tokens):
                if input_ids[i + j] != token:
                    match = False
                    break
            if match:
                cot_end_pos = i + len(thought_end_tokens) - 1
                break

        if cot_end_pos != -1:
            cot_start_pos = 0
            json_start_pos = cot_end_pos + 1
            json_end_pos = len(input_ids) - 1

        return cot_start_pos, cot_end_pos, json_start_pos, json_end_pos

    def _construct_user_prompt(self, data):
        """Build user prompt from data fields using f-string."""
        return f"""
##基础画像
-profile_1：{data.get('profile_feat_1', '')}
-profile_2：{data.get('profile_feat_2', '')}
-profile_3：{data.get('profile_feat_3', '')}
-profile_4：{data.get('profile_feat_4', '')}
-profile_5：{data.get('profile_feat_5', '')}
-profile_6：{data.get('profile_feat_6', '')}{data.get('profile_feat_7', '')}
-profile_8：{data.get('profile_feat_8', '')}{data.get('profile_feat_9', '')}
-profile_10：{data.get('profile_feat_10', '')}
-profile_11：{data.get('profile_feat_11', '')}
-profile_12：{data.get('profile_feat_12', '')}
-profile_13：{data.get('profile_feat_13', '')}
-profile_14：{data.get('profile_feat_14', '')}

##历史行为
-短期行为序列：{data.get('short_term_behavior_seq', '')}
-长期行为：{data.get('long_term_behavior', '')}

#当前上下文
##时空信息
-当前时间：{data.get('currenttime', '')}
-是否周末：{data.get('weekendflag', '')}
-是否节假日：{data.get('holidayflag', '')}
-当前城市：{data.get('current_city', '')}{data.get('current_district', '')}
-当前POI名称：{data.get('current_poi_name', '')}
-当前POI类型：{data.get('current_tag', '')}
##事件信息
-trigger_1：{data.get('trigger_1', '')}
-trigger_2：{data.get('trigger_2', '')}{data.get('trigger_3', '')}{data.get('trigger_4', '')}
-trigger_5：{data.get('trigger_5', '')}
-trigger_6：{data.get('trigger_6', '')}
-trigger_7：{data.get('trigger_7', '')}
"""

    def _process_func(self, messages, tokenizer):
        """Convert messages into input_ids / labels / token_weights / token_types for model training."""
        chat_template_sig = inspect.signature(tokenizer.apply_chat_template)
        supports_thinking = 'enable_thinking' in chat_template_sig.parameters

        template_kwargs = dict(tokenize=False, add_generation_prompt=False)
        if supports_thinking:
            template_kwargs['enable_thinking'] = False

        full_text = tokenizer.apply_chat_template(messages, **template_kwargs)

        prompt_messages = messages[:-1]
        prompt_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if supports_thinking:
            prompt_kwargs['enable_thinking'] = False
        prompt_text = tokenizer.apply_chat_template(prompt_messages, **prompt_kwargs)

        full = tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        prompt = tokenizer(
            prompt_text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )

        input_ids = full.input_ids[0]
        attention_mask = full.attention_mask[0]
        labels = input_ids.clone()
        full_len = attention_mask.sum().item()

        assistant_marker = "<|im_start|>assistant\n"
        assistant_marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)

        prompt_len = None
        input_ids_list = input_ids.tolist()
        for i in range(len(input_ids_list) - len(assistant_marker_ids) + 1):
            if input_ids_list[i:i + len(assistant_marker_ids)] == assistant_marker_ids:
                prompt_len = i + len(assistant_marker_ids)
                break

        if prompt_len is None:
            prompt_len = prompt.attention_mask[0].sum().item()

        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # token_weights and token_types are used by WeightedLossTrainer
        token_weights = torch.ones_like(labels, dtype=torch.float32)
        token_types = torch.zeros_like(labels, dtype=torch.long)  # 0=other, 1=CoT, 2=JSON

        assistant_start = prompt_len
        assistant_ids = input_ids[assistant_start:full_len]
        assistant_text = messages[2]['content'] if len(messages) > 2 else ""

        cot_start, cot_end, json_start, json_end = self._find_cot_json_boundary(
            assistant_ids, assistant_text, tokenizer
        )

        if cot_start != -1 and cot_end != -1:
            cot_abs_start = assistant_start + cot_start
            cot_abs_end = assistant_start + cot_end
            token_weights[cot_abs_start:cot_abs_end + 1] = self.cot_weight
            token_types[cot_abs_start:cot_abs_end + 1] = 1

        if json_start != -1 and json_end != -1:
            json_abs_start = assistant_start + json_start
            json_abs_end = assistant_start + json_end
            token_weights[json_abs_start:json_abs_end + 1] = self.json_weight
            token_types[json_abs_start:json_abs_end + 1] = 2

        if self.epoch != self.last_printed_epoch:
            print(f"[Epoch {self.epoch}] CoT=[{cot_start},{cot_end}], JSON=[{json_start},{json_end}], "
                  f"cot_weight={self.cot_weight}, json_weight={self.json_weight}")
            decoded = tokenizer.decode(input_ids[labels != -100], skip_special_tokens=False)
            print(f"[Epoch {self.epoch}] Learning target preview: {decoded[:200]}...")
            self.last_printed_epoch = self.epoch

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_weights": token_weights,
            "token_types": token_types
        }

    def __call__(self, csv_row):
        """Process one CSV row and return a dict or messages for model training."""
        data = self._parse_csv_row(csv_row)
        if not data:
            return self._get_empty_sample()

        clean_label = self._clean_assistant_response(data.get('raw_labels'))
        if not clean_label:
            return self._get_empty_sample()

        user_content = self._construct_user_prompt(data)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": clean_label}
        ]

        if not self.applied_tokenizer:
            return {"messages": messages}

        return self._process_func(messages, self.tokenizer)
