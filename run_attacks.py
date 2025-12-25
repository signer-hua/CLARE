# -*- coding: utf-8 -*-
"""
批量运行对抗攻击实验 - 直接运行版本
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import torch
import time
import pickle
from tqdm import tqdm
import copy
import re

# 将当前目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
try:
    from bertattack_fraud import (
        FraudDialogueFeature,
        Classifier,
        Tokenizer,
        clean_dialogue_for_attack,
        _tokenize_chinese_dialogue,
        get_important_scores_fraud,
        attack_fraud_dialogue,
        evaluate_fraud_attack
    )
except ImportError as e:
    print(f"❌ 无法导入模块: {e}")
    print("请确保 bertattack_fraud.py 在同一目录下")
    sys.exit(1)


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


def run_single_attack_experiment(data_path, model_path, model_type, output_dir,
                                 k=20, batch_size=32, max_length=128,
                                 threshold_pred_score=0.3, max_change_rate=0.4,
                                 start_idx=0, end_idx=50, device=None):
    """
    运行单个攻击实验
    """
    print(f"\n🔧 准备攻击实验: {model_path}")

    # 创建实验输出目录
    model_name = os.path.basename(model_path).replace('.pt', '').replace('.pth', '')
    exp_dir = os.path.join(output_dir, f"attack_{model_type}_{model_name}")
    os.makedirs(exp_dir, exist_ok=True)

    # 自动检测设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"⚡ 使用设备: {device}")

    # 加载模型
    try:
        checkpoint = torch.load(model_path, map_location=device)

        if model_type == "bert":
            # 导入BERT相关模块
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            from bertattack_fraud import BertClassifier

            tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

            # 创建模型结构
            model = BertClassifier('bert-base-chinese', 2)
            model.load_state_dict(checkpoint['model_state_dict'])

            # 加载MLM模型
            mlm_model = AutoModelForMaskedLM.from_pretrained('bert-base-chinese')
            mlm_model.to(device)
            mlm_model.eval()

        else:  # base model
            # 加载tokenizer
            tokenizer_path = os.path.join(os.path.dirname(model_path), 'tokenizer.pkl')
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer = pickle.load(f)
            else:
                print(f"⚠️  未找到tokenizer文件，创建默认tokenizer")
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
            mlm_model = None

        model.to(device)
        model.eval()

        print(f"✅ 模型加载成功 (验证准确率: {checkpoint.get('val_acc', '未知'):.4f})")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 加载数据
    features_raw = get_fraud_data_cls(data_path)
    if not features_raw:
        print("❌ 数据加载失败")
        return None

    if end_idx is None or end_idx > len(features_raw):
        end_idx = len(features_raw)

    features_to_attack = features_raw[start_idx:end_idx]
    print(f"🎯 攻击范围: {start_idx} 到 {end_idx} (共 {len(features_to_attack)} 条)")

    # 统计标签分布
    labels = [label for _, label in features_to_attack]
    fraud_count = sum(labels)
    normal_count = len(labels) - fraud_count
    print(f"📊 样本分布: 欺诈 {fraud_count} 条, 正常 {normal_count} 条")

    # 执行攻击
    print(f"⚡ 开始对抗攻击...")
    attacked_features = []

    start_time = time.time()

    with torch.no_grad():
        for i, (seq, label) in enumerate(tqdm(features_to_attack, desc="攻击进度", unit="条")):
            try:
                feature = FraudDialogueFeature(seq, label)
                feature = attack_fraud_dialogue(
                    feature,
                    model,
                    mlm_model,
                    tokenizer,
                    k,
                    batch_size,
                    device,
                    max_length=max_length,
                    use_bpe=0,
                    threshold_pred_score=threshold_pred_score,
                    max_change_rate=max_change_rate
                )
                attacked_features.append(feature)

            except Exception as e:
                print(f"⚠️  第 {i + 1} 条样本攻击失败: {e}")
                failed_feature = FraudDialogueFeature(seq, label)
                failed_feature.success = 0
                failed_feature.final_adverse = seq
                attacked_features.append(failed_feature)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"✅ 攻击完成! 总耗时: {total_time:.2f}秒")

    # 评估结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(exp_dir, f"attack_results_{timestamp}.json")
    stats_json = os.path.join(exp_dir, f"attack_stats_{timestamp}.json")

    print(f"📈 评估攻击效果...")
    stats = evaluate_fraud_attack(attacked_features, output_json)

    # 保存统计信息
    with open(stats_json, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 保存实验日志
    log_file = os.path.join(exp_dir, "experiment_log.txt")
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"实验时间: {timestamp}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"模型类型: {model_type}\n")
        f.write(f"数据路径: {data_path}\n")
        f.write(f"样本数量: {len(features_to_attack)}\n")
        f.write(f"攻击参数: k={k}, batch_size={batch_size}, max_length={max_length}\n")
        f.write(f"总耗时: {total_time:.2f}秒\n\n")
        f.write(f"攻击成功率: {stats.get('attack_success_rate', 0) * 100:.2f}%\n")
        f.write(f"原始准确率: {stats.get('original_accuracy', 0) * 100:.2f}%\n")
        f.write(f"攻击后准确率: {stats.get('after_attack_accuracy', 0) * 100:.2f}%\n")

    print(f"💾 实验文件:")
    print(f"  详细结果: {output_json}")
    print(f"  统计信息: {stats_json}")
    print(f"  实验日志: {log_file}")

    return {
        'success': True,
        'model_type': model_type,
        'model_path': model_path,
        'output_dir': exp_dir,
        'result_file': output_json,
        'stats_file': stats_json,
        'log_file': log_file,
        'stats': stats
    }


def find_models(model_dir):
    """
    查找所有可用的模型
    """
    models = []

    # 1. 查找BERT模型
    bert_model_path = os.path.join(model_dir, "bert_fraud_classifier", "best_model.pt")
    if os.path.exists(bert_model_path):
        print(f"✅ 找到BERT模型: {bert_model_path}")
        models.append(("bert", bert_model_path))
    else:
        print(f"⚠️  未找到BERT模型: {bert_model_path}")

    # 2. 查找分类器模型
    base_model_path = os.path.join(model_dir, "classifier", "best_model.pt")
    if os.path.exists(base_model_path):
        print(f"✅ 找到分类器模型: {base_model_path}")
        models.append(("base", base_model_path))
    else:
        print(f"⚠️  未找到分类器模型: {base_model_path}")

    # 3. 查找其他可能存在的模型
    model_patterns = ["*.pt", "*.pth", "*.ckpt"]
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file.endswith(('.pt', '.pth', '.ckpt')):
                model_path = os.path.join(root, file)
                if model_path not in [m[1] for m in models]:
                    # 尝试判断模型类型
                    if 'bert' in file.lower() or 'bert' in root.lower():
                        models.append(("bert", model_path))
                        print(f"✅ 找到其他BERT模型: {model_path}")
                    else:
                        models.append(("base", model_path))
                        print(f"✅ 找到其他分类器模型: {model_path}")

    return models


def collect_results(results, output_dir):
    """
    收集所有实验结果
    """
    print("\n📊 收集实验结果...")

    summary = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiments': [],
        'summary_stats': {}
    }

    successful_experiments = []

    for result in results:
        if result and result.get('success'):
            exp_info = {
                'model_type': result['model_type'],
                'model_path': result['model_path'],
                'model_name': os.path.basename(result['model_path']),
                'output_dir': result['output_dir'],
                'result_file': result['result_file'],
                'stats': result.get('stats', {})
            }

            summary['experiments'].append(exp_info)
            successful_experiments.append(exp_info)

            stats = result.get('stats', {})
            success_rate = stats.get('attack_success_rate', 0) * 100
            print(f"  ✅ {result['model_type']} - {os.path.basename(result['model_path'])}: "
                  f"成功率 {success_rate:.1f}%")
        else:
            model_path = result.get('model_path', '未知') if result else '未知'
            model_type = result.get('model_type', '未知') if result else '未知'
            print(f"  ❌ 攻击失败 {model_type} - {os.path.basename(model_path)}")

    # 汇总统计
    if successful_experiments:
        models = [f"{exp['model_type']} - {exp['model_name']}" for exp in successful_experiments]
        success_rates = [exp['stats'].get('attack_success_rate', 0) for exp in successful_experiments]
        after_accuracies = [exp['stats'].get('after_attack_accuracy', 0) for exp in successful_experiments]
        original_accuracies = [exp['stats'].get('original_accuracy', 0) for exp in successful_experiments]

        summary['summary_stats'] = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_experiments),
            'models': models,
            'success_rates': success_rates,
            'after_attack_accuracies': after_accuracies,
            'original_accuracies': original_accuracies,
            'avg_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
            'avg_after_accuracy': sum(after_accuracies) / len(after_accuracies) if after_accuracies else 0,
            'avg_original_accuracy': sum(original_accuracies) / len(original_accuracies) if original_accuracies else 0
        }

    # 保存汇总
    summary_file = os.path.join(output_dir, f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 创建CSV报告
    if successful_experiments:
        csv_data = []
        for exp in successful_experiments:
            stats = exp['stats']
            csv_data.append({
                'Model Type': exp['model_type'].upper(),
                'Model': exp['model_name'],
                'Attack Success Rate': f"{stats.get('attack_success_rate', 0) * 100:.2f}%",
                'After Attack Accuracy': f"{stats.get('after_attack_accuracy', 0) * 100:.2f}%",
                'Original Accuracy': f"{stats.get('original_accuracy', 0) * 100:.2f}%",
                'Avg Queries': f"{stats.get('avg_queries', 0):.1f}",
                'Avg Change Rate': f"{stats.get('avg_change_rate', 0) * 100:.2f}%",
                'Success Count': stats.get('success_count', 0),
                'Total Samples': stats.get('total_samples', 0)
            })

        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, f"results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')

        print(f"\n💾 CSV报告已保存到: {csv_file}")
        print("\n📈 攻击结果汇总:")
        print(df.to_string(index=False))

    return summary


def main():
    """
    批量攻击实验主函数
    """
    print("=" * 60)
    print("欺诈对话对抗攻击批量实验系统")
    print("=" * 60)

    # ========== 硬编码参数配置 ==========
    # 在这里修改参数，然后直接运行

    # 数据路径
    DATA_PATH = "data/processed/fraud_test_small.txt"  # 小样本测试
    # DATA_PATH = "data/processed/fraud_test.txt"      # 完整测试集

    # 模型目录
    MODEL_DIR = "models"

    # 输出目录
    OUTPUT_DIR = "./experiments"

    # 实验配置
    SAMPLE_SIZE = 50  # 每个实验的样本数
    START_IDX = 0  # 起始索引

    # 攻击参数
    K = 20  # Top-K候选词
    BATCH_SIZE = 32  # 批量大小
    MAX_LENGTH = 128  # 最大文本长度
    THRESHOLD_PRED_SCORE = 0.3
    MAX_CHANGE_RATE = 0.4

    # ========== 主程序开始 ==========

    print("\n📋 配置参数:")
    print(f"  数据路径: {DATA_PATH}")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  样本数量: {SAMPLE_SIZE}")
    print(f"  Top-K: {K}, 批量大小: {BATCH_SIZE}")

    # 检查数据文件
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 数据文件不存在: {DATA_PATH}")
        print("请先运行 preprocess_fraud.py 预处理数据")
        return

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(OUTPUT_DIR, f"batch_experiment_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n📁 实验目录: {exp_dir}")

    # 查找所有可用模型
    print("\n🔍 查找可用模型...")
    models = find_models(MODEL_DIR)

    if not models:
        print("\n❌ 未找到任何可用的模型！")
        print("请先运行以下命令训练模型:")
        print("1. 训练分类器模型: python train_classifiers.py")
        print("2. 训练BERT模型: python train_bert.py (如果需要)")
        return

    print(f"\n✅ 找到 {len(models)} 个模型")

    # 运行所有攻击实验
    print(f"\n⚡ 开始批量攻击实验 (共 {len(models)} 个模型)")
    results = []

    for i, (model_type, model_path) in enumerate(models):
        print(f"\n{'=' * 50}")
        print(f"实验 {i + 1}/{len(models)}: {model_type.upper()} - {os.path.basename(model_path)}")
        print(f"{'=' * 50}")

        # 计算样本范围
        start = START_IDX
        end = start + SAMPLE_SIZE

        result = run_single_attack_experiment(
            data_path=DATA_PATH,
            model_path=model_path,
            model_type=model_type,
            output_dir=exp_dir,
            k=K,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            threshold_pred_score=THRESHOLD_PRED_SCORE,
            max_change_rate=MAX_CHANGE_RATE,
            start_idx=start,
            end_idx=end
        )

        results.append(result)

    # 收集结果
    print(f"\n{'=' * 50}")
    print("📊 收集实验结果...")
    print(f"{'=' * 50}")

    summary = collect_results(results, exp_dir)

    print(f"\n{'=' * 50}")
    print("🎉 批量实验完成！")
    print(f"📁 结果目录: {exp_dir}")
    print(f"{'=' * 50}")

    # 显示关键结果
    if summary.get('summary_stats'):
        stats = summary['summary_stats']
        print(f"\n📈 关键统计:")
        print(f"  平均攻击成功率: {stats['avg_success_rate'] * 100:.1f}%")
        print(f"  平均攻击后准确率: {stats['avg_after_accuracy'] * 100:.1f}%")
        print(f"  平均原始准确率: {stats['avg_original_accuracy'] * 100:.1f}%")
        print(f"  成功实验数: {stats['successful_experiments']}/{stats['total_experiments']}")

    print("\n🔍 查看详细结果:")
    print(f"  1. 打开目录: {exp_dir}")
    print(f"  2. 查看CSV报告文件")
    print(f"  3. 查看各实验的详细日志")

    print(f"\n📋 下一步:")
    print(f"  1. 分析对抗样本以了解攻击方式")
    print(f"  2. 调整攻击参数重新实验")
    print(f"  3. 训练更多模型进行对比")


if __name__ == "__main__":
    main()