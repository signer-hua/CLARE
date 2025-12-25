# -*- coding: utf-8 -*-
"""
文本分类器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import os
import re
from collections import Counter
import pickle


class Tokenizer:

    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'

    def build_vocab(self, texts):
        """构建词汇表"""
        # 统计所有字符
        char_counter = Counter()
        for text in texts:
            char_counter.update(text)

        # 选择最常见的字符
        most_common = char_counter.most_common(self.vocab_size - 4)  # 为特殊token留位置

        # 构建映射
        self.char_to_idx = {self.unk_token: 0, self.pad_token: 1, self.cls_token: 2, self.sep_token: 3}
        self.idx_to_char = {0: self.unk_token, 1: self.pad_token, 2: self.cls_token, 3: self.sep_token}

        for idx, (char, _) in enumerate(most_common, start=4):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char

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


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoded = self.tokenizer.encode(text, self.max_length)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(data_path):
    """加载数据"""
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
                text = parts[0].strip()
                if text:
                    texts.append(text)
                    try:
                        labels.append(int(parts[1]))
                    except:
                        labels.append(1)

        print(f"✅ 加载了 {len(texts)} 条样本")

    except Exception as e:
        print(f"❌ 加载失败: {e}")

    return texts, labels


def train_model():
    """训练模型"""
    print("=" * 60)
    print("🤖 文本分类器训练")
    print("=" * 60)

    # 配置参数
    TRAIN_DATA = "data/processed/fraud_train.txt"
    VAL_DATA = "data/processed/fraud_val.txt"
    OUTPUT_DIR = "models/classifier"

    VOCAB_SIZE = 5000
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    MAX_LENGTH = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据
    print("\n📊 加载数据...")
    train_texts, train_labels = load_data(TRAIN_DATA)

    if os.path.exists(VAL_DATA):
        val_texts, val_labels = load_data(VAL_DATA)
    else:
        # 分割验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
        )

    # 限制样本数量用于快速训练
    SAMPLE_LIMIT = 1000
    if len(train_texts) > SAMPLE_LIMIT:
        print(f"🔧 采样 {SAMPLE_LIMIT} 条样本进行训练...")
        train_texts = train_texts[:SAMPLE_LIMIT]
        train_labels = train_labels[:SAMPLE_LIMIT]

    if len(val_texts) > 200:
        val_texts = val_texts[:200]
        val_labels = val_labels[:200]

    print(f"\n📈 数据统计:")
    print(f"  训练集: {len(train_texts)} 条")
    print(f"    欺诈: {sum(train_labels)} 条 ({sum(train_labels) / len(train_labels) * 100:.1f}%)")
    print(
        f"    正常: {len(train_labels) - sum(train_labels)} 条 ({(len(train_labels) - sum(train_labels)) / len(train_labels) * 100:.1f}%)")

    print(f"\n  验证集: {len(val_texts)} 条")
    print(f"    欺诈: {sum(val_labels)} 条 ({sum(val_labels) / len(val_labels) * 100:.1f}%)")
    print(
        f"    正常: {len(val_labels) - sum(val_labels)} 条 ({(len(val_labels) - sum(val_labels)) / len(val_labels) * 100:.1f}%)")

    # 创建tokenizer并构建词汇表
    print("\n🔧 构建词汇表...")
    tokenizer = Tokenizer(VOCAB_SIZE)
    tokenizer.build_vocab(train_texts)
    print(f"✅ 词汇表大小: {len(tokenizer.char_to_idx)}")

    # 保存tokenizer
    with open(os.path.join(OUTPUT_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)

    # 创建数据集
    print("📦 创建数据集...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

    # 创建模型
    print("\n🏗️  创建模型...")
    model = Classifier(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=2
    )

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"⚡ 使用设备: {device}")

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    print("\n🚀 开始训练...")
    best_val_acc = 0

    for epoch in range(EPOCHS):
        print(f"\n📅 Epoch {epoch + 1}/{EPOCHS}")

        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0

        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_true, val_preds)

        print(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}, 验证准确率: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'vocab_size': VOCAB_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'hidden_dim': HIDDEN_DIM,
            }, os.path.join(OUTPUT_DIR, 'best_model.pt'))
            print(f"  💾 保存最佳模型 (准确率: {val_acc:.4f})")

    # 最终评估
    print("\n🎯 最终评估...")

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_true, all_preds)
    final_f1 = f1_score(all_true, all_preds, average='weighted')

    print(f"\n📊 最终结果:")
    print(f"  验证准确率: {final_acc:.4f}")
    print(f"  F1分数: {final_f1:.4f}")

    print("\n📋 分类报告:")
    print(classification_report(all_true, all_preds, target_names=['正常', '欺诈'], digits=4))

    print("\n🔢 混淆矩阵:")
    cm = confusion_matrix(all_true, all_preds)
    print(f"[[TN FP]\n [FN TP]] = \n{cm}")

    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print("=" * 60)

    print(f"\n📁 模型保存到: {OUTPUT_DIR}")
    print(f"💾 最佳模型: {OUTPUT_DIR}/best_model.pt")
    print(f"💾 Tokenizer: {OUTPUT_DIR}/tokenizer.pkl")

    return model, tokenizer


if __name__ == "__main__":
    train_model()