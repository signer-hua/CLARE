# -*- coding: utf-8 -*-
"""
处理欺诈对话数据的脚本
分别处理训练和测试文件
"""

import pandas as pd
import re
import json
import os
import sys
import numpy as np
from pathlib import Path
import random


def clean_dialogue_text(text):
    """
    清洗对话文本，移除角色标记和多余格式
    """
    if not isinstance(text, str):
        return ""

    # 移除"音频内容："前缀
    if text.startswith("音频内容："):
        text = text[5:]

    # 移除left:/right:角色标记
    text = re.sub(r'(left:|right:)\s*', '', text)

    # 移除多余的星号、换行和空白
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # 移除对话标记和引号
    text = re.sub(r'【.*?】', '', text)
    text = text.replace('"', '').replace("'", "")

    return text.strip()


def process_csv_file(file_path, file_type="train"):
    """
    处理单个CSV文件
    """
    print(f"\n📋 处理{file_type}文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return None

    # 读取CSV文件
    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ 使用编码: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            print(f"❌ 无法读取文件 {file_path}")
            return None
    except Exception as e:
        print(f"❌ 读取文件失败 - {e}")
        return None

    print(f"📊 数据形状: {df.shape}")
    print(f"📝 列名: {df.columns.tolist()}")

    # 自动检测文本列和标签列
    text_col = None
    label_col = None

    # 查找文本列
    text_keywords = ['text', 'content', 'dialogue', '对话', '文本', '内容', 'specific_dialogue_content']
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in text_keywords:
            if keyword in col_lower:
                text_col = col
                print(f"✅ 找到文本列: {text_col}")
                break
        if text_col:
            break

    if not text_col:
        # 如果没有找到，使用第一列
        text_col = df.columns[0]
        print(f"⚠️  未找到文本列，使用第一列: {text_col}")

    # 查找标签列
    label_keywords = ['label', 'fraud', '诈骗', '欺诈', 'is_fraud', 'is_fraudulent']
    for col in df.columns:
        col_lower = str(col).lower()
        for keyword in label_keywords:
            if keyword in col_lower:
                label_col = col
                print(f"✅ 找到标签列: {label_col}")
                break
        if label_col:
            break

    if not label_col:
        print(f"❌ 未找到标签列")
        return None

    # 显示数据预览
    print(f"\n📄 数据预览（前3行）:")
    for i in range(min(3, len(df))):
        text_preview = str(df.iloc[i][text_col])
        if len(text_preview) > 50:
            text_preview = text_preview[:50] + "..."

        label_val = df.iloc[i][label_col]
        print(f"  行 {i}: 文本={text_preview}, 标签={label_val}")

    # 清洗文本
    print("🧹 清洗对话文本...")
    df['cleaned_text'] = df[text_col].apply(clean_dialogue_text)

    # 转换标签
    print("🏷️  转换标签...")

    def convert_label(x):
        if pd.isna(x):
            return 1  # 默认欺诈
        x_str = str(x).upper().strip()
        if x_str in ['TRUE', 'T', '1', '是', 'YES', 'Y', '欺诈', 'FRAUD', '客服诈骗', '银行诈骗']:
            return 1
        elif x_str in ['FALSE', 'F', '0', '否', 'NO', 'N', '正常', 'NORMAL']:
            return 0
        else:
            # 尝试转换为数字
            try:
                val = float(x)
                return 1 if val > 0.5 else 0
            except:
                return 1  # 默认欺诈

    df['label'] = df[label_col].apply(convert_label)

    print(f"✅ {file_type}数据处理完成: {len(df)} 条记录")
    print(f"   欺诈样本: {sum(df['label'])} 条 ({sum(df['label']) / len(df) * 100:.1f}%)")
    print(f"   正常样本: {len(df) - sum(df['label'])} 条 ({(len(df) - sum(df['label'])) / len(df) * 100:.1f}%)")

    return df


def save_bert_format(data, filename, output_dir):
    """
    保存为BERT-Attack格式
    """
    output_path = output_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("text_a\tlabel\n")
        for _, row in data.iterrows():
            text = row['cleaned_text']
            label = row['label']
            if text and len(text.strip()) > 0:  # 只保存非空文本
                f.write(f"{text}\t{label}\n")

    return output_path


def main():
    """主函数 - 分别处理训练和测试文件"""
    print("=" * 60)
    print("🎯 欺诈对话数据预处理系统 ")
    print("(分别处理训练和测试文件)")
    print("=" * 60)

    print(f"当前目录: {Path.cwd()}")
    print(f"Python版本: {sys.version}")

    # ========== 硬编码文件路径 ==========
    TRAIN_FILE = "data/train_result.csv"  # 训练文件
    TEST_FILE = "data/test_result.csv"  # 测试文件

    # ========== 创建输出目录 ==========
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 处理训练文件 ==========
    print(f"\n{'=' * 50}")
    print("处理训练文件")
    print(f"{'=' * 50}")

    if not os.path.exists(TRAIN_FILE):
        print(f"❌ 训练文件不存在: {TRAIN_FILE}")
        print("请将 train_result.csv 放在 data/ 目录下")
        return

    train_df = process_csv_file(TRAIN_FILE, "训练")
    if train_df is None or len(train_df) == 0:
        print("❌ 训练数据处理失败")
        return

    # ========== 处理测试文件 ==========
    print(f"\n{'=' * 50}")
    print("处理测试文件")
    print(f"{'=' * 50}")

    if not os.path.exists(TEST_FILE):
        print(f"❌ 测试文件不存在: {TEST_FILE}")
        print("请将 test_result.csv 放在 data/ 目录下")
        return

    test_df = process_csv_file(TEST_FILE, "测试")
    if test_df is None or len(test_df) == 0:
        print("❌ 测试数据处理失败")
        return

    # ========== 保存文件 ==========
    print(f"\n{'=' * 50}")
    print("保存处理后的文件")
    print(f"{'=' * 50}")

    # 保存训练集
    train_path = save_bert_format(train_df, "fraud_train.txt", output_dir)
    print(f"✅ 训练集已保存: {train_path} ({len(train_df)} 条)")

    # 保存测试集
    test_path = save_bert_format(test_df, "fraud_test.txt", output_dir)
    print(f"✅ 测试集已保存: {test_path} ({len(test_df)} 条)")

    # 创建小测试集（用于快速测试）
    test_small = test_df.head(min(100, len(test_df)))
    test_small_path = save_bert_format(test_small, "fraud_test_small.txt", output_dir)
    print(f"✅ 小测试集已保存: {test_small_path} ({len(test_small)} 条)")

    # 创建验证集（从训练集中分割）
    print(f"\n📊 从训练集中分割验证集...")
    from sklearn.model_selection import train_test_split

    # 分割训练集为训练和验证
    train_texts = train_df['cleaned_text'].tolist()
    train_labels = train_df['label'].tolist()

    train_texts_new, val_texts, train_labels_new, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    # 创建验证集DataFrame
    val_df = pd.DataFrame({
        'cleaned_text': val_texts,
        'label': val_labels
    })

    val_path = save_bert_format(val_df, "fraud_val.txt", output_dir)
    print(f"✅ 验证集已保存: {val_path} ({len(val_df)} 条)")

    # 更新训练集
    train_df_new = pd.DataFrame({
        'cleaned_text': train_texts_new,
        'label': train_labels_new
    })

    # 覆盖原来的训练集
    train_path = save_bert_format(train_df_new, "fraud_train.txt", output_dir)
    print(f"✅ 更新后的训练集已保存: {train_path} ({len(train_df_new)} 条)")

    # ========== 数据统计 ==========
    print(f"\n{'=' * 50}")
    print("📊 最终数据统计")
    print(f"{'=' * 50}")

    print(f"训练集: {len(train_df_new)} 条")
    train_fraud = sum(train_df_new['label'])
    train_normal = len(train_df_new) - train_fraud
    print(f"  欺诈: {train_fraud} 条 ({train_fraud / len(train_df_new) * 100:.1f}%)")
    print(f"  正常: {train_normal} 条 ({train_normal / len(train_df_new) * 100:.1f}%)")

    print(f"\n验证集: {len(val_df)} 条")
    val_fraud = sum(val_df['label'])
    val_normal = len(val_df) - val_fraud
    print(f"  欺诈: {val_fraud} 条 ({val_fraud / len(val_df) * 100:.1f}%)")
    print(f"  正常: {val_normal} 条 ({val_normal / len(val_df) * 100:.1f}%)")

    print(f"\n测试集: {len(test_df)} 条")
    test_fraud = sum(test_df['label'])
    test_normal = len(test_df) - test_fraud
    print(f"  欺诈: {test_fraud} 条 ({test_fraud / len(test_df) * 100:.1f}%)")
    print(f"  正常: {test_normal} 条 ({test_normal / len(test_df) * 100:.1f}%)")

    print(f"\n小测试集: {len(test_small)} 条")
    small_fraud = sum(test_small['label'])
    small_normal = len(test_small) - small_fraud
    print(f"  欺诈: {small_fraud} 条 ({small_fraud / len(test_small) * 100:.1f}%)")
    print(f"  正常: {small_normal} 条 ({small_normal / len(test_small) * 100:.1f}%)")

    print("\n" + "=" * 60)
    print("🎉 数据预处理完成！")
    print("=" * 60)



if __name__ == "__main__":
    main()