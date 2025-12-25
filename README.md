# 欺诈对话检测与对抗攻击系统 (Fraud Dialogue Detection & Adversarial Attack)

## 📖 项目简介

本项目是一个针对中文电信欺诈对话的检测与对抗攻击评估系统。主要功能包括数据预处理、构建深度学习分类器（BiLSTM）识别欺诈对话，以及利用对抗攻击算法（基于 BERT-Attack 改进）生成对抗样本，从而评估和测试模型的鲁棒性。

## 📂 项目结构

```text
demo/
├── data/                   # 数据存放目录
│   ├── processed/          # 预处理后的 TXT 数据 (train/test/val)
│   ├── train_result.csv    # 原始训练数据
│   └── test_result.csv     # 原始测试数据
├── models/                 # 模型保存目录
│   └── classifier/         # LSTM 分类器模型及 tokenizer
├── experiments/            # 批量攻击实验结果目录
├── results/                # 单次攻击测试结果
├── preprocess_fraud.py     # 数据预处理脚本
├── train_classifiers.py    # LSTM 分类模型训练脚本
├── bertattack_fraud.py     # 对抗攻击核心算法实现
├── run_attacks.py          # 批量攻击实验运行脚本
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.x 及以下依赖库：
- torch
- pandas
- numpy
- scikit-learn
- transformers
- tqdm

### 2. 数据预处理
首先处理原始 CSV 数据，清洗文本并划分为训练集、验证集和测试集。

```bash
python preprocess_fraud.py
```
> 输出：将在 `data/processed/` 目录下生成 `fraud_train.txt`, `fraud_test.txt` 等文件。

### 3. 训练分类模型
训练一个基于 BiLSTM 的文本分类器来识别欺诈对话。

```bash
python train_classifiers.py
```
> 输出：模型将保存在 `models/classifier/`，包含 `best_model.pt` 和 `tokenizer.pkl`。

### 4. 运行对抗攻击
对训练好的模型进行对抗攻击测试，生成对抗样本并评估攻击成功率。

**方式 A：批量运行实验 (推荐)**
自动扫描可用模型并运行批量测试。
```bash
python run_attacks.py
```

**方式 B：单次调试**
直接运行攻击核心脚本进行调试。
```bash
python bertattack_fraud.py
```

## 📊 核心模块说明

### 数据预处理 (`preprocess_fraud.py`)
- **清洗逻辑**：移除 "音频内容：" 前缀、角色标记 (left/right)、多余符号及格式。
- **标签转换**：统一将各种形式的标签（True/False, 1/0, 汉字）转换为标准的 0/1 格式。

### 模型架构 (`train_classifiers.py`)
- **Tokenizer**：自定义字符级分词器，构建字表。
- **Classifier**：
  - Embedding 层 (128维)
  - BiLSTM 层 (双向, 隐层256维)
  - 全连接层 + Dropout
  - 输出 2 分类 (正常/欺诈)

### 对抗攻击 (`bertattack_fraud.py`)
基于参考论文的 **CLARE (Contextualized Perturbation)** 思路对中文对话进行适配，核心是“上下文约束下的三类编辑操作 + 贪心搜索”：
1. **重要性排序**：使用 leave-one-out masking 计算每个位置对目标模型预测的重要性。
2. **三类扰动操作（CLARE）**：
   - **Replace（替换）**：对当前位置进行 Masked LM 预测，用上下文相关候选替换原字符/词。
   - **Insert（插入）**：在当前位置前插入一个 `[MASK]`，用 Masked LM 生成插入候选。
   - **Merge（合并/删除）**：通过删除当前位置字符实现“merge/压缩”式编辑（对话场景下用于更轻量的结构扰动）。
3. **攻击策略**：每轮在候选操作中选择使“原预测标签概率下降最大”的扰动，迭代直到预测翻转或达到最大改动率。

模型差异：
- **BERT 模型**：Replace/Insert 使用 Masked LM 生成上下文相关候选。
- **BiLSTM（字符级）模型**：在不依赖 MLM 的情况下，使用简化候选（形近字/同类字符、常见语气助词插入、删除）实现同样的三操作搜索流程。

## 📈 结果分析
实验结果将保存在 `experiments/` 目录下，包含：
- `attack_results_*.json`: 详细的攻击案例（原始文本 vs 对抗文本）。
- `results_summary_*.csv`: 汇总统计（攻击成功率、修改率、查询次数等）。

提示：新版攻击的 `attack_results_*.json` 中 `changes` 会记录 `op` 字段（`replace/insert/merge`），用于区分每一步采用的 CLARE 操作类型。
