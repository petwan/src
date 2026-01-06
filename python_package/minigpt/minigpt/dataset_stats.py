# minigpt/dataset_stats.py

import json
import argparse
from tokenizer import Tokenizer
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np


def get_num_tokens(file_path: str, tokenizer: Tokenizer) -> List[int]:
    """
    读取 JSONL 文件，对每条样本的 question + answer 进行 tokenize，
    返回每个样本的 token 数量列表。
    """
    input_num_tokens = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    question = data.get("question", "")
                    answer = data.get("answer", "")
                    tokens, _ = tokenizer.encode(
                        question, answer, pad_to_max_length=False
                    )
                    input_num_tokens.append(len(tokens))
                except json.JSONDecodeError:
                    print(
                        f"⚠️ 第 {line_num} 行 JSON 解析失败，跳过。内容: {line[:50]}..."
                    )
                except KeyError as e:
                    print(f"⚠️ 第 {line_num} 行缺少字段 {e}，跳过。")
    except FileNotFoundError:
        print(f"❌ 错误：文件 {file_path} 不存在！")
        raise SystemExit(1)
    return input_num_tokens


def count_intervals(
    num_tokens: List[int], interval: int = 20, max_length: int = None
) -> Dict[str, int]:
    """按区间统计 token 长度分布"""
    if not num_tokens:
        return {}

    actual_max = max(num_tokens)
    upper_bound = min(actual_max, max_length) if max_length else actual_max

    intervals_count = {}
    current = 0
    while current <= upper_bound:
        next_bound = current + interval
        count = sum(1 for x in num_tokens if current <= x < next_bound)
        intervals_count[f"{current}-{next_bound}"] = count
        current = next_bound

    if max_length and actual_max > max_length:
        overflow_count = sum(1 for x in num_tokens if x >= max_length)
        intervals_count[f">{max_length}"] = overflow_count

    return intervals_count


def plot_token_distribution(
    num_tokens: List[int],
    intervals_count: Dict[str, int],
    interval: int = 20,
    max_length: int = 512,
    title: str = "Token 分布",
):
    """绘制并显示 token 长度分布柱状图，并标出 90% 分位竖线"""
    if not intervals_count or not num_tokens:
        print("📊 无数据可绘图。")
        return

    # 计算 95% 分位数
    p95 = np.percentile(num_tokens, 95)

    # 构建数值 x 坐标和对应的标签
    x_vals = []
    labels = []
    y_vals = []

    current = 0
    # 添加常规区间
    while True:
        key = f"{current}-{current + interval}"
        if key in intervals_count:
            x_vals.append(current + interval / 2)  # 区间中点
            labels.append(key)
            y_vals.append(intervals_count[key])
            current += interval
        else:
            break

    # 添加溢出区间（如果有）
    overflow_key = f">{max_length}"
    if overflow_key in intervals_count:
        # 将溢出区间放在 max_length 右侧一点，避免重叠
        x_vals.append(max_length + interval / 2)
        labels.append(overflow_key)
        y_vals.append(intervals_count[overflow_key])

    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制柱状图
    bars = ax.bar(
        x_vals, y_vals, width=interval * 0.8, color="lightcoral", edgecolor="black"
    )

    # 设置 x 轴
    ax.set_xticks(x_vals)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # 画 90% 分位竖线
    ax.axvline(
        p95,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"95% Percentile ({p95:.1f})",
    )

    # 添加柱子顶部的数值标签
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # 设置标题和标签
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Sample Nbr", fontsize=12)
    ax.set_xlabel("Token Length", fontsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="分析训练集 Token 长度分布")
    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="训练数据路径（JSONL 格式，每行一个 {'question': ..., 'answer': ...}）",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="词表文件路径（用于初始化 Tokenizer）",
    )
    parser.add_argument(
        "--interval", type=int, default=20, help="统计区间步长（默认: 20）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大统计长度，超过的归入 '>max_length' 区间（默认: 512）",
    )
    parser.add_argument(
        "--no_plot", action="store_true", help="不显示图表（仅打印统计结果）"
    )

    args = parser.parse_args()

    # 初始化分词器
    try:
        tokenizer = Tokenizer(args.vocab_path)
    except Exception as e:
        print(f"❌ 分词器初始化失败: {e}")
        raise SystemExit(1)

    # 获取 token 长度
    print(f"🔍 正在处理数据: {args.train_data}")
    num_tokens = get_num_tokens(args.train_data, tokenizer)
    print(f"✅ 共加载 {len(num_tokens)} 条有效样本。")

    if not num_tokens:
        print("❌ 未找到有效样本，退出。")
        return

    # 统计分布
    intervals = count_intervals(
        num_tokens, interval=args.interval, max_length=args.max_length
    )

    # 打印结果
    print("\n📊 Token 长度分布统计:")
    total = 0
    for interval_label, count in intervals.items():
        print(f"  {interval_label:>12}: {count:>6}")
        total += count
    print(f"{'-' * 20}\n  总计: {total}")

    # 可选：绘图（现在传入原始 num_tokens）
    if not args.no_plot:
        plot_token_distribution(
            num_tokens=num_tokens,
            intervals_count=intervals,
            interval=args.interval,
            max_length=args.max_length,
            title=f"Token Length Distribution ({args.train_data})",
        )


if __name__ == "__main__":
    main()
