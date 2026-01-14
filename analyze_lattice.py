import torch
import yaml
import numpy as np
from collections import Counter
import os
import pandas as pd
def calculate_cell_lengths(pt_file_path):
    """
    读取 .pt 文件，提取晶胞矩阵，并计算晶胞的边长。
    
    Args:
        pt_file_path: .pt 文件的路径，应该包含一个包含多个晶体数据的列表。
    
    Returns:
        stats_dict: 包含晶胞边长统计数据的字典。
    """
    # 加载 .pt 文件，假设它是一个包含多个晶体数据的列表
    crystal_data_list = torch.load(pt_file_path, weights_only=False)

    if not isinstance(crystal_data_list, list):
        raise TypeError(f"Expected data from {pt_file_path} to be a list, but got {type(crystal_data_list)}")

    # 存储晶胞边长的列表
    cell_lengths = []

    for i, crystal_data_dict in enumerate(crystal_data_list):
        # 提取晶胞矩阵
        lengths = crystal_data_dict['graph_arrays']['lengths']
        cell_lengths.append(lengths)

    if not cell_lengths:
        raise ValueError("No valid crystal data found in the .pt file.")

    # 计算晶胞长度的统计指标
    cell_lengths = np.array(cell_lengths)
    
    # 计算每个方向的边长统计
    a_lengths = cell_lengths[:, 0]
    b_lengths = cell_lengths[:, 1]
    c_lengths = cell_lengths[:, 2]

    stats_dict = {
        "a_lengths": {
            "mean": np.mean(a_lengths),
            "std": np.std(a_lengths),
            "min": np.min(a_lengths),
            "max": np.max(a_lengths),
            "histogram": np.histogram(a_lengths, bins=20),  # 使用20个bins
            "data": a_lengths,
        },
        "b_lengths": {
            "mean": np.mean(b_lengths),
            "std": np.std(b_lengths),
            "min": np.min(b_lengths),
            "max": np.max(b_lengths),
            "histogram": np.histogram(b_lengths, bins=20),
            "data": b_lengths,
        },
        "c_lengths": {
            "mean": np.mean(c_lengths),
            "std": np.std(c_lengths),
            "min": np.min(c_lengths),
            "max": np.max(c_lengths),
            "histogram": np.histogram(c_lengths, bins=20),
            "data": c_lengths,
        },
        "overall": {
            "mean": np.mean([a_lengths, b_lengths, c_lengths], axis=1).mean(),
            "std": np.std([a_lengths, b_lengths, c_lengths], axis=1).mean(),
            "min": min(np.min(a_lengths), np.min(b_lengths), np.min(c_lengths)),
            "max": max(np.max(a_lengths), np.max(b_lengths), np.max(c_lengths)),
        }
    }

    # 保存统计数据到 YAML 文件
    stats_yaml_path = pt_file_path.replace('.pt', '_cell_stats.yaml')
    with open(stats_yaml_path, 'w') as f:
        yaml.dump(stats_dict, f)

    return stats_dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def plot_cell_length_stats(stats_dict, save_dir):
    """
    使用 Matplotlib 和 Seaborn 绘制并保存晶胞边长的统计图表。
    
    Args:
        stats_dict: 从 `calculate_cell_lengths` 函数返回的统计字典。
        save_dir: 保存图像的目录路径。
    """
    # 设置绘图样式
    sns.set(style="whitegrid")

    # 提取每个方向的晶胞边长和对应的频率
    a_lengths_edges, a_freq = np.array(stats_dict["a_lengths"]["histogram"][1]), np.array(stats_dict["a_lengths"]["histogram"][0])
    b_lengths_edges, b_freq = np.array(stats_dict["b_lengths"]["histogram"][1]), np.array(stats_dict["b_lengths"]["histogram"][0])
    c_lengths_edges, c_freq = np.array(stats_dict["c_lengths"]["histogram"][1]), np.array(stats_dict["c_lengths"]["histogram"][0])

    # 计算区间中点
    a_centers = (a_lengths_edges[:-1] + a_lengths_edges[1:]) / 2  # 计算区间中心
    b_centers = (b_lengths_edges[:-1] + b_lengths_edges[1:]) / 2
    c_centers = (c_lengths_edges[:-1] + c_lengths_edges[1:]) / 2

    # 创建 DataFrame 来确保长度和频率一一对应
    a_data = pd.DataFrame({
        'Length': a_centers.round(1),
        'Frequency': a_freq
    })

    b_data = pd.DataFrame({
        'Length': b_centers.round(1),
        'Frequency': b_freq
    })

    c_data = pd.DataFrame({
        'Length': c_centers.round(1),
        'Frequency': c_freq
    })

    # 创建保存图表的目录（如果不存在的话）
    os.makedirs(save_dir, exist_ok=True)

    # 设置图形大小

    # 绘制 a_lengths 的条形图
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Length', y='Frequency', data=a_data, color='skyblue')
    plt.title("Distribution of a_lengths")
    plt.xlabel("a_length (Å)")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_dir}/a_lengths_distribution.png")  # 保存为图片
    plt.clf()  # 清除当前图形，避免与下一个图形重叠

    # 绘制 b_lengths 的条形图
    sns.barplot(x='Length', y='Frequency', data=b_data, color='lightgreen')
    plt.title("Distribution of b_lengths")
    plt.xlabel("b_length (Å)")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_dir}/b_lengths_distribution.png")  # 保存为图片
    plt.clf()  # 清除当前图形

    # 绘制 c_lengths 的条形图
    sns.barplot(x='Length', y='Frequency', data=c_data, color='salmon')
    plt.title("Distribution of c_lengths")
    plt.xlabel("c_length (Å)")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_dir}/c_lengths_distribution.png")  # 保存为图片
    plt.clf()  # 清除当前图形

    # --- 绘制箱线图 ---
    # 将各方向晶胞边长放在同一张图中，方便比较
    cell_lengths_data = [stats_dict["a_lengths"]["data"], stats_dict["b_lengths"]["data"], stats_dict["c_lengths"]["data"]]  # 使用去掉最后一个边界的部分
    labels = ["a_lengths", "b_lengths", "c_lengths"]

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=cell_lengths_data)
    plt.xticks(np.arange(3), labels=labels)
    plt.title("Boxplot of Cell Lengths")
    plt.ylabel("Length (Å)")
    plt.savefig(f"{save_dir}/cell_lengths_boxplot.png")  # 保存为图片
    plt.clf()  # 清除当前图形

# 假设你已经从 pt 文件中读取并计算了统计数据
pt_file_path = 'data/processed_mp_20_train.pt'
stats = calculate_cell_lengths(pt_file_path)
plot_cell_length_stats(stats, '.')

