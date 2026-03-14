import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def analyze_dataset(data_path):
    """分析数据集的基本统计信息"""
    print("=" * 50)
    print("数据集分析报告")
    print("=" * 50)

    # 读取数据 - 修改这里
    # CSV格式，有表头
    df = pd.read_csv(data_path)

    # 查看列名
    print(f"\n0. 数据格式")
    print(f"   列名: {df.columns.tolist()}")
    print(f"   原始数据量: {len(df)}")

    # 确保列名正确（可能是'up'和'down'，也可能是其他）
    # 假设第一列是上联，第二列是下联
    if len(df.columns) >= 2:
        input_col = df.columns[0]  # 第一列是上联
        target_col = df.columns[1]  # 第二列是下联
        print(f"   上联列: '{input_col}', 下联列: '{target_col}'")
    else:
        print("   错误: 数据列数不足")
        return

    # 处理空值
    df = df.dropna(subset=[input_col, target_col])
    print(f"   删除空值后: {len(df)}")

    # 确保数据是字符串类型
    df[input_col] = df[input_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    # 过滤掉空字符串
    df = df[df[input_col].str.len() > 0]
    df = df[df[target_col].str.len() > 0]
    print(f"   过滤空字符串后: {len(df)}")

    # 基本统计
    print(f"\n1. 基本统计")
    print(f"   总条数: {len(df)}")
    print(f"   唯一上联数: {df[input_col].nunique()}")
    print(f"   唯一上联比例: {df[input_col].nunique() / len(df):.2%}")
    print(f"   唯一上联比例: {df[target_col].nunique() / len(df):.2%}")

    # 长度统计
    df['input_len'] = df[input_col].apply(len)
    df['target_len'] = df[target_col].apply(len)

    print(f"\n2. 长度统计")
    print(f"   上联平均长度: {df['input_len'].mean():.2f}")
    print(f"   下联平均长度: {df['target_len'].mean():.2f}")
    print(f"   上联最大长度: {df['input_len'].max()}")
    print(f"   上联最小长度: {df['input_len'].min()}")
    print(f"   下联最大长度: {df['target_len'].max()}")
    print(f"   下联最小长度: {df['target_len'].min()}")

    # 长度分布
    print(f"\n3. 长度分布（主要范围）")
    len_dist = df['input_len'].value_counts().sort_index()
    # 只显示出现次数>1000的长度
    len_dist_filtered = len_dist[len_dist > 1000]
    for length, count in len_dist_filtered.head(15).items():
        print(f"   长度 {length}: {count} 条 ({count / len(df):.2%})")

    # 字数统计
    all_chars = []
    for text in df[input_col]:
        all_chars.extend(list(text))
    for text in df[target_col]:
        all_chars.extend(list(text))

    char_counter = Counter(all_chars)
    print(f"\n4. 字符统计")
    print(f"   总字符数: {sum(char_counter.values()):,}")
    print(f"   唯一字符数: {len(char_counter)}")
    print(f"   最常用10个字符: {char_counter.most_common(10)}")

    # 数据质量检查
    print(f"\n5. 数据质量检查")
    # 检查上下联长度是否相等
    equal_len = (df['input_len'] == df['target_len']).sum()
    print(f"   上下联长度相等: {equal_len} 条 ({equal_len / len(df):.2%})")

    return {
        'total': len(df),
        'unique_input': df[input_col].nunique(),
        'unique_target': df[target_col].nunique(),
        'avg_input_len': df['input_len'].mean(),
        'avg_target_len': df['target_len'].mean(),
        'max_len': max(df['input_len'].max(), df['target_len'].max()),
        'min_len': min(df['input_len'].min(), df['target_len'].min()),
        'vocab_size': len(char_counter)
    }