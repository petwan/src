# split_data.py
import json
import argparse
import random
from pathlib import Path


def split_jsonl_data(input_path: str, train_ratio: float = 0.9, seed: int = 42):
    """
    将 JSONL 格式的 QA 数据集划分为 train.jsonl 和 val.jsonl。

    Args:
        input_path (str): 原始数据路径（JSONL）
        train_ratio (float): 训练集比例（0.0 ～ 1.0）
        seed (int): 随机种子，确保可复现
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # 读取所有有效行
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "question" in item and "answer" in item:
                    data.append(line)  # 保留原始字符串，避免格式变化
                else:
                    print(
                        f"⚠️  Warning: Line {line_num} missing 'question' or 'answer', skipped."
                    )
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Line {line_num} is invalid JSON, skipped.")

    if not data:
        raise ValueError("No valid data found!")

    # 打乱并划分
    random.seed(seed)
    random.shuffle(data)
    n_train = int(len(data) * train_ratio)

    train_data = data[:n_train]
    val_data = data[n_train:]

    # 输出路径
    output_dir = input_path.parent
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    # 写入文件
    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_data) + "\n")

    print("✅ Split completed!")
    print(f"   Total samples: {len(data)}")
    print(f"   Train: {len(train_data)} → {train_path}")
    print(f"   Val:   {len(val_data)} → {val_path}")
    print(f"   Train ratio: {train_ratio:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split QA dataset into train/val sets."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to original JSONL dataset (e.g., data/all.jsonl)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Proportion of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    split_jsonl_data(args.input, args.train_ratio, args.seed)
