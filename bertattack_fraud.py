# -*- coding: utf-8 -*-
"""
对抗攻击脚本 - 适配模型版本
支持两种模型：BERT模型和分类器模型
"""

import warnings
import os
import sys
import torch
import torch.nn as nn
import json
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModel
import copy
import numpy as np
from tqdm import tqdm
import time
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)

# 停用词列表（针对中文对话优化）
filter_words = ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
                '说', '要', '去', '你',
                '会', '着', '没有', '看', '好', '自己', '这', '那', '他', '她', '它', '我们', '你们', '他们', '这个',
                '那个', '什么', '怎么',
                '为什么', '吗', '呢', '吧', '啊', '呀', '哦', '嗯', '呃', '然后', '但是', '可是', '不过', '而且',
                '所以', '因为', '如果',
                '虽然', '即使', '既然', '为了', '关于', '对于', '根据', '按照', '通过', '随着', '作为', '而且', '或者',
                '还是', '不仅', '而且',
                '既', '又', '无论', '不管', '尽管', '即使', '假如', '倘若', '只要', '只有', '除非', '无论', '不论',
                '不管', '尽管', '即使',
                '既然', '因为', '所以', '因此', '于是', '然后', '接着', '最后', '首先', '其次', '再次', '另外', '此外',
                '同时', '同样',
                '相反', '反而', '然而', '可是', '但是', '不过', '只是', '却是', '倒是', '就是', '都是', '总是', '又是',
                '还是', '也是',
                '就是', '就是', '就是', '就是', '就是', '就是', '就是', '就是', '就是', '就是']

filter_words = set(filter_words)


class FraudDialogueFeature:
    """
    欺诈对话特征类，扩展原始Feature类
    """

    def __init__(self, seq_a, label, original_info=None):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0  # 0:失败, 1:替换过多, 2:未找到对抗样本, 3:原始错误, 4:攻击成功
        self.sim = 0.0
        self.changes = []
        self.original_info = original_info  # 保存原始信息
        self.attack_type = "word_replacement"  # 攻击类型


class BertClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese', num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class Classifier(nn.Module):
    """分类器"""

    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_classes=2):
        super(Classifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None):
        # 嵌入层
        embedded = self.embedding(input_ids)

        # LSTM
        lstm_out, _ = self.lstm(embedded)

        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]

        # 分类
        output = self.dropout(last_hidden)
        logits = self.fc(output)

        return logits


class Tokenizer:
    """字符级tokenizer"""

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'

    def encode(self, text, max_length=128):
        """编码文本"""
        tokens = [self.char_to_idx.get(char, self.char_to_idx[self.unk_token]) for char in text[:max_length - 2]]

        # 添加特殊token
        tokens = [self.char_to_idx[self.cls_token]] + tokens + [self.char_to_idx[self.sep_token]]

        # 填充
        if len(tokens) < max_length:
            tokens = tokens + [self.char_to_idx[self.pad_token]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]
            tokens[-1] = self.char_to_idx[self.sep_token]

        # 创建attention mask
        attention_mask = [1 if token != self.char_to_idx[self.pad_token] else 0 for token in tokens]

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }


def clean_dialogue_for_attack(text):
    """
    攻击前的对话清洗
    """
    if not isinstance(text, str):
        return ""

    # 移除特殊标记和多余空格
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text


def get_fraud_data_cls(data_path):
    """
    加载欺诈对话数据
    """
    print(f"📂 加载数据: {data_path}")

    texts = []
    labels = []

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 跳过表头
        if lines[0].strip() == 'text_a\tlabel':
            lines = lines[1:]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                text = clean_dialogue_for_attack(parts[0])
                if text:
                    texts.append(text)
                    try:
                        labels.append(int(parts[1]))
                    except:
                        labels.append(1)

        print(f"✅ 加载了 {len(texts)} 条样本")

    except Exception as e:
        print(f"❌ 加载失败: {e}")

    return list(zip(texts, labels))


def _tokenize_chinese_dialogue(seq, tokenizer):
    """
    中文对话分词处理
    """
    seq = seq.replace('\n', '').replace('\t', ' ')

    # 字符级分词（对于中文更合适）
    words = list(seq)

    sub_words = []
    keys = []
    index = 0

    for word in words:
        # 对于Tokenizer，直接返回字符
        if isinstance(tokenizer, Tokenizer):
            sub = [word]
        else:
            # BERT tokenizer
            sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)

    return words, sub_words, keys


def get_important_scores_fraud(words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length,
                               device):
    """
    计算词重要性分数
    """
    masked_words = []

    # 创建掩码版本
    for i in range(len(words)):
        masked = words.copy()
        masked[i] = '[UNK]' if not isinstance(tokenizer, Tokenizer) else tokenizer.unk_token
        masked_words.append(masked)

    # 准备输入
    texts = [''.join(words) for words in masked_words]

    all_input_ids = []
    all_attention_masks = []

    for text in texts:
        if isinstance(tokenizer, Tokenizer):
            inputs = tokenizer.encode(text, max_length)
            all_input_ids.append(inputs['input_ids'].unsqueeze(0))
            all_attention_masks.append(inputs['attention_mask'].unsqueeze(0))
        else:
            inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            all_input_ids.append(inputs['input_ids'])
            all_attention_masks.append(inputs['attention_mask'])

    seqs = torch.cat(all_input_ids, dim=0).to(device)
    masks = torch.cat(all_attention_masks, dim=0).to(device)

    # 批量计算概率
    leave_1_probs = []
    for i in range(0, len(seqs), batch_size):
        batch_seqs = seqs[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]

        with torch.no_grad():
            outputs = tgt_model(batch_seqs, batch_masks)
            leave_1_prob_batch = torch.softmax(outputs, -1)
            leave_1_probs.append(leave_1_prob_batch)

    if leave_1_probs:
        leave_1_probs = torch.cat(leave_1_probs, dim=0)
    else:
        leave_1_probs = torch.zeros(len(words), orig_probs.size(-1)).to(device)

    leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

    # 计算重要性分数
    import_scores = (
            orig_prob
            - leave_1_probs[:, orig_label]
            + (leave_1_probs_argmax != orig_label).float()
            * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
    ).data.cpu().numpy()

    return import_scores


def _predict_with_target_model(text, tgt_model, tokenizer, device, max_length):
    if isinstance(tokenizer, Tokenizer):
        inputs = tokenizer.encode(text, max_length)
        input_ids = inputs['input_ids'].unsqueeze(0).to(device)
        attention_mask = inputs['attention_mask'].unsqueeze(0).to(device)
    else:
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = tgt_model(input_ids, attention_mask)
        probs = torch.softmax(outputs, -1).squeeze()

    return probs


def _is_valid_mlm_token(token, tokenizer):
    if token is None:
        return False
    token = str(token)
    if token in {tokenizer.unk_token, tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token}:
        return False
    if token.startswith('##'):
        return False
    if token.startswith('[') and token.endswith(']'):
        return False
    if token.strip() == '':
        return False
    return True


def _mlm_candidates_for_single_mask(text_with_mask, tokenizer, mlm_model, device, k, max_length):
    if mlm_model is None:
        return []

    inputs = tokenizer(
        text_with_mask,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    mask_id = tokenizer.mask_token_id
    mask_positions = (input_ids[0] == mask_id).nonzero(as_tuple=False)
    if mask_positions.numel() == 0:
        return []

    mask_index = int(mask_positions[0].item())

    with torch.no_grad():
        outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, mask_index, :]
        probs = torch.softmax(logits, -1)

    top_probs, top_ids = torch.topk(probs, k)
    tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    return [(tok, float(score)) for tok, score in zip(tokens, top_probs.tolist())]


def _generate_candidates_clare(text, idx, tokenizer, mlm_model, k, max_length):
    words = list(text)
    if idx < 0 or idx >= len(words):
        return []

    original_char = words[idx]
    masked_text = ''.join(words[:idx]) + tokenizer.mask_token + ''.join(words[idx + 1:])
    raw = _mlm_candidates_for_single_mask(masked_text, tokenizer, mlm_model, device=next(mlm_model.parameters()).device,
                                          k=k, max_length=max_length)
    candidates = []
    for tok, score in raw:
        if not _is_valid_mlm_token(tok, tokenizer):
            continue
        if tok == original_char:
            continue
        if tok in filter_words:
            continue
        if len(tok) != 1:
            continue
        candidates.append((tok, score))
    return candidates


def _generate_insert_candidates_clare(text, idx, tokenizer, mlm_model, k, max_length):
    words = list(text)
    if idx < 0 or idx > len(words):
        return []

    masked_text = ''.join(words[:idx]) + tokenizer.mask_token + ''.join(words[idx:])
    raw = _mlm_candidates_for_single_mask(masked_text, tokenizer, mlm_model, device=next(mlm_model.parameters()).device,
                                          k=k, max_length=max_length)
    candidates = []
    for tok, score in raw:
        if not _is_valid_mlm_token(tok, tokenizer):
            continue
        if tok in filter_words:
            continue
        if len(tok) != 1:
            continue
        candidates.append((tok, score))
    return candidates


def attack_fraud_dialogue(feature, tgt_model, mlm_model, tokenizer, k, batch_size, device,
                          max_length=512, use_bpe=0, threshold_pred_score=0.3, max_change_rate=0.4):
    """
    针对欺诈对话的攻击函数
    """
    feature.attack_type = "clare"

    original_text = feature.seq
    current_text = original_text

    if not isinstance(current_text, str) or current_text.strip() == '':
        feature.success = 2
        return feature

    orig_probs = _predict_with_target_model(current_text, tgt_model, tokenizer, device, max_length)
    feature.query += 1
    orig_label = int(torch.argmax(orig_probs).item())
    current_label = orig_label

    if orig_label != feature.label:
        feature.success = 3
        return feature

    max_changes = int(max_change_rate * max(1, len(original_text)))

    similar_chars = {
        '我': ['你', '他', '她'],
        '你': ['我', '他', '她'],
        '他': ['她', '它', '你'],
        '她': ['他', '它', '你'],
        '是': ['否', '非'],
        '不': ['没', '勿'],
        '有': ['无', '没'],
        '没': ['不', '无', '未'],
        '钱': ['款', '金', '资'],
        '卡': ['号', '证', '账'],
        '转': ['汇', '打', '付'],
        '账': ['款', '费', '金']
    }

    while feature.change < max_changes:
        words = list(current_text)
        if len(words) == 0:
            break

        current_probs = _predict_with_target_model(current_text, tgt_model, tokenizer, device, max_length)
        feature.query += 1
        current_label = int(torch.argmax(current_probs).item())
        if current_label != orig_label:
            feature.final_adverse = current_text
            feature.success = 4
            return feature

        current_orig_prob = float(current_probs[orig_label].item())

        important_scores = get_important_scores_fraud(
            words, tgt_model, current_probs.max(), orig_label, current_probs,
            tokenizer, batch_size, max_length, device
        )
        feature.query += len(words)

        ranked = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
        candidate_positions = [idx for idx, _ in ranked[:min(20, len(ranked))]]

        best = None

        for idx in candidate_positions:
            tgt_char = words[idx]
            if tgt_char in filter_words:
                continue
            if tgt_char.strip() == '':
                continue

            replace_candidates = []
            insert_candidates_before = []

            if isinstance(tokenizer, Tokenizer):
                replace_candidates = [(c, 1.0) for c in similar_chars.get(tgt_char, []) if c != tgt_char]
                insert_candidates_before = [(c, 1.0) for c in ['啊', '呢', '吧', '呀', '哦'] if c not in filter_words]
            else:
                replace_candidates = _generate_candidates_clare(current_text, idx, tokenizer, mlm_model, k, max_length)
                insert_candidates_before = _generate_insert_candidates_clare(current_text, idx, tokenizer, mlm_model, k,
                                                                            max_length)

            for cand, _ in replace_candidates[:10]:
                new_text = ''.join(words[:idx]) + cand + ''.join(words[idx + 1:])
                probs = _predict_with_target_model(new_text, tgt_model, tokenizer, device, max_length)
                feature.query += 1
                new_label = int(torch.argmax(probs).item())
                if new_label != orig_label:
                    feature.change += 1
                    feature.changes.append({
                        'op': 'replace',
                        'position': idx,
                        'original': tgt_char,
                        'replacement': cand,
                        'success': True
                    })
                    feature.final_adverse = new_text
                    feature.success = 4
                    return feature
                gap = current_orig_prob - float(probs[orig_label].item())
                if gap > 0:
                    if best is None or gap > best['gap']:
                        best = {
                            'op': 'replace',
                            'position': idx,
                            'original': tgt_char,
                            'replacement': cand,
                            'gap': gap,
                            'new_text': new_text
                        }

            for cand, _ in insert_candidates_before[:10]:
                new_text = ''.join(words[:idx]) + cand + ''.join(words[idx:])
                probs = _predict_with_target_model(new_text, tgt_model, tokenizer, device, max_length)
                feature.query += 1
                new_label = int(torch.argmax(probs).item())
                if new_label != orig_label:
                    feature.change += 1
                    feature.changes.append({
                        'op': 'insert',
                        'position': idx,
                        'original': '',
                        'replacement': cand,
                        'success': True
                    })
                    feature.final_adverse = new_text
                    feature.success = 4
                    return feature
                gap = current_orig_prob - float(probs[orig_label].item())
                if gap > 0:
                    if best is None or gap > best['gap']:
                        best = {
                            'op': 'insert',
                            'position': idx,
                            'original': '',
                            'replacement': cand,
                            'gap': gap,
                            'new_text': new_text
                        }

            if idx < len(words) - 1:
                new_text = ''.join(words[:idx]) + ''.join(words[idx + 1:])
            else:
                new_text = ''.join(words[:idx])

            if new_text != current_text and new_text.strip() != '':
                probs = _predict_with_target_model(new_text, tgt_model, tokenizer, device, max_length)
                feature.query += 1
                new_label = int(torch.argmax(probs).item())
                if new_label != orig_label:
                    feature.change += 1
                    feature.changes.append({
                        'op': 'merge',
                        'position': idx,
                        'original': tgt_char,
                        'replacement': '',
                        'success': True
                    })
                    feature.final_adverse = new_text
                    feature.success = 4
                    return feature
                gap = current_orig_prob - float(probs[orig_label].item())
                if gap > 0:
                    if best is None or gap > best['gap']:
                        best = {
                            'op': 'merge',
                            'position': idx,
                            'original': tgt_char,
                            'replacement': '',
                            'gap': gap,
                            'new_text': new_text
                        }

        if best is None:
            break

        feature.change += 1
        feature.changes.append({
            'op': best['op'],
            'position': best['position'],
            'original': best['original'],
            'replacement': best['replacement'],
            'success': False,
            'gap': float(best['gap'])
        })
        current_text = best['new_text']

    feature.final_adverse = current_text
    feature.success = 2
    return feature


def evaluate_fraud_attack(features, output_json=None):
    """
    欺诈对话攻击评估
    """
    print("\n" + "=" * 60)
    print("📊 欺诈对话对抗攻击评估结果")
    print("=" * 60)

    total = len(features)
    success_count = 0
    original_error = 0
    total_queries = 0
    total_changes = 0
    total_words = 0

    success_features = []

    for feat in features:
        total_words += len(feat.seq)

        if feat.success == 3:
            original_error += 1
        elif feat.success == 4:
            success_count += 1
            total_queries += feat.query
            total_changes += feat.change
            success_features.append(feat)

    # 计算指标
    if success_count > 0:
        avg_queries = total_queries / success_count
        avg_change_rate = total_changes / total_words if total_words > 0 else 0
    else:
        avg_queries = 0
        avg_change_rate = 0

    original_accuracy = 1 - (original_error / total) if total > 0 else 0
    attack_success_rate = success_count / (total - original_error) if (total - original_error) > 0 else 0
    after_attack_accuracy = 1 - attack_success_rate

    print(f"总样本数: {total}")
    print(f"原始预测错误: {original_error} ({original_error / total * 100:.2f}%)")
    print(f"攻击成功: {success_count} ({success_count / total * 100:.2f}%)")
    print(f"攻击失败: {total - original_error - success_count}")
    print(f"\n攻击前准确率: {original_accuracy:.4f}")
    print(f"攻击后准确率: {after_attack_accuracy:.4f}")
    print(f"攻击成功率: {attack_success_rate:.4f}")
    print(f"平均查询次数: {avg_queries:.2f}")
    print(f"平均改动率: {avg_change_rate:.4f}")

    # 保存成功案例
    if output_json and success_features:
        output_data = []
        for feat in success_features:
            output_data.append({
                'original_text': feat.seq,
                'adversarial_text': feat.final_adverse,
                'label': int(feat.label),
                'query_times': int(feat.query),
                'changes': feat.changes,
                'change_count': int(feat.change),
                'text_length': len(feat.seq)
            })

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 成功案例已保存到: {output_json}")

    return {
        'total_samples': total,
        'original_accuracy': original_accuracy,
        'after_attack_accuracy': after_attack_accuracy,
        'attack_success_rate': attack_success_rate,
        'avg_queries': avg_queries,
        'avg_change_rate': avg_change_rate,
        'success_count': success_count
    }


def main():
    """
    主函数 - 支持两种模型版本
    """
    print("=" * 60)
    print("🔧 欺诈对话对抗攻击系统 (支持BERT和分类器模型)")
    print("=" * 60)

    # ========== 硬编码参数配置 ==========
    DATA_PATH = "data/processed/fraud_test_small.txt"  # 小样本测试
    MODEL_TYPE = "base"  # "bert" 或 "base"

    if MODEL_TYPE == "bert":
        TGT_PATH = "models/bert_fraud_classifier/best_model.pt"  # BERT模型路径
        MLM_PATH = "bert-base-chinese"  # MLM模型
    else:
        TGT_PATH = "models/classifier/best_model.pt"  # 分类器模型路径
        MLM_PATH = None  # 分类器模型不需要MLM

    OUTPUT_DIR = "./results"
    K = 20
    BATCH_SIZE = 32
    MAX_LENGTH = 128
    THRESHOLD_PRED_SCORE = 0.3
    MAX_CHANGE_RATE = 0.4
    START_IDX = 0
    END_IDX = 50

    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚡ 使用设备: {device}")

    # ========== 主程序开始 ==========

    print(f"📂 数据路径: {DATA_PATH}")
    print(f"🎯 目标模型: {TGT_PATH}")
    print(f"🤖 模型类型: {MODEL_TYPE}")

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查文件
    if not os.path.exists(DATA_PATH):
        print(f"❌ 数据文件不存在: {DATA_PATH}")
        return

    if not os.path.exists(TGT_PATH):
        print(f"❌ 目标模型不存在: {TGT_PATH}")
        if MODEL_TYPE == "bert":
            print("请先运行 train_bert.py 训练BERT模型")
        else:
            print("请先运行 train_classifiers.py 训练分类器模型")
        return

    # 加载模型
    print("\n⏳ 加载模型中...")

    # 加载目标模型
    try:
        checkpoint = torch.load(TGT_PATH, map_location=device)

        if MODEL_TYPE == "bert":
            # BERT模型
            from transformers import AutoTokenizer as BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

            # 创建模型结构
            model = BertClassifier('bert-base-chinese', 2)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 加载MLM模型
            from transformers import BertForMaskedLM
            mlm_model = BertForMaskedLM.from_pretrained(MLM_PATH)
            mlm_model.to(device)
            mlm_model.eval()
            print(f"✅ MLM模型加载成功")

        else:
            # 分类器模型
            # 加载tokenizer
            tokenizer_path = os.path.join(os.path.dirname(TGT_PATH), 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                print(f"⚠️  未找到tokenizer文件，使用默认配置")
                tokenizer = Tokenizer()

            # 创建模型结构
            vocab_size = checkpoint.get('vocab_size', 5000)
            embedding_dim = checkpoint.get('embedding_dim', 128)
            hidden_dim = checkpoint.get('hidden_dim', 256)

            model = Classifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_classes=2
            )
            model.load_state_dict(checkpoint['model_state_dict'])

            # 分类器模型不需要MLM
            mlm_model = None

        model.to(device)
        model.eval()

        print(f"✅ 目标模型加载成功 (验证准确率: {checkpoint.get('val_acc', '未知'):.4f})")

    except Exception as e:
        print(f"❌ 目标模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 加载数据
    print("\n⏳ 加载数据中...")
    features_raw = get_fraud_data_cls(DATA_PATH)

    if not features_raw:
        print("❌ 数据加载失败")
        return

    if END_IDX is None or END_IDX > len(features_raw):
        END_IDX = len(features_raw)

    features_to_attack = features_raw[START_IDX:END_IDX]
    print(f"🎯 攻击范围: {START_IDX} 到 {END_IDX} (共 {len(features_to_attack)} 条)")

    # 统计标签分布
    labels = [label for _, label in features_to_attack]
    fraud_count = sum(labels)
    normal_count = len(labels) - fraud_count
    print(f"📊 样本分布: 欺诈 {fraud_count} 条, 正常 {normal_count} 条")

    # 执行攻击
    print(f"\n⚡ 开始对抗攻击...")
    attacked_features = []

    start_time = time.time()

    with torch.no_grad():
        for i, (seq, label) in enumerate(tqdm(features_to_attack, desc="攻击进度", unit="条")):
            try:
                feature = FraudDialogueFeature(seq, label)
                feature = attack_fraud_dialogue(
                    feature,
                    model,  # 目标模型
                    mlm_model,
                    tokenizer,
                    K,
                    BATCH_SIZE,
                    device,  # 传入设备参数
                    max_length=MAX_LENGTH,
                    use_bpe=0,
                    threshold_pred_score=THRESHOLD_PRED_SCORE,
                    max_change_rate=MAX_CHANGE_RATE
                )
                attacked_features.append(feature)

            except Exception as e:
                print(f"⚠️  第 {i + 1} 条样本攻击失败: {e}")
                import traceback
                traceback.print_exc()
                failed_feature = FraudDialogueFeature(seq, label)
                failed_feature.success = 0
                failed_feature.final_adverse = seq
                attacked_features.append(failed_feature)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n✅ 攻击完成! 总耗时: {total_time:.2f}秒")

    # 评估结果
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(OUTPUT_DIR, f"attack_results_{MODEL_TYPE}_{timestamp}.json")
    stats_json = os.path.join(OUTPUT_DIR, f"attack_stats_{MODEL_TYPE}_{timestamp}.json")

    print(f"\n📈 评估攻击效果...")
    stats = evaluate_fraud_attack(attacked_features, output_json)

    # 保存统计信息
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n💾 结果文件:")
    print(f"  详细结果: {output_json}")
    print(f"  统计信息: {stats_json}")
    print("=" * 60)
    print(f"🎉 {MODEL_TYPE.upper()}模型对抗攻击实验完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
