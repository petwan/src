import torch
from torch.utils.data import Dataset
import json
from minigpt.tokenizer import Tokenizer


class QADataset(Dataset):
    """
    自回归训练用的问答数据集（Question-Answer Dataset）。

    将 (question, answer) 拼接为单个序列，并构造：
      - input_ids:   [SOS, q1, q2, ..., <sep>, a1, a2, ..., <sep>]
      - targets:     [q1, q2, ..., <sep>, a1, a2, ..., <sep>, EOS]

    实际通过 shift 实现：input = tokens[:-1], target = tokens[1:]
    """

    def __init__(self, data_path: str, tokenizer: Tokenizer, max_length: int):
        self.tokeniner = tokenizer
        self.max_length = max_length
        self.data = []

        print(f"Loading data from {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            # 逐行读取数据, 行数从1开始
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if "question" not in item or "answer" not in item:
                        print(
                            f"⚠️  Line {line_num}: Missing 'question' or 'answer', skipped."
                        )
                        continue
                    self.data.append((item["question"], item["answer"]))

                except Exception as e:
                    print(f"⚠️  Line {line_num}: Invalid JSON, skipped. Error: {e}")

        print(f"✅ Loaded {len(self.data)} valid QA pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question, answer = self.data[idx]
        # encode question and answer
        full_tokens, atnn_mask = self.tokeniner.encode(
            question, answer, max_length=self.max_length
        )

        # 自回归训练： input 向右移一位，target 向左移一位
        input_ids = full_tokens[:-1]  # 去掉最后一个 token
        attention_mask = atnn_mask[:-1]  # 对应 input_ids 的 attention mask
        targets = full_tokens[1:]  # 去掉第一个 token

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.long),
        }


if __name__ == "__main__":
    tokenizer = Tokenizer("./data/vocab.json")
    dataset = QADataset("./data/train.jsonl", tokenizer, max_length=128)

    print(f"数据集大小：{len(dataset)}")

    item = dataset[0]
    print(item)
    print(tokenizer.decode(item["input_ids"].tolist()))
    print(tokenizer.decode(item["targets"].tolist()))
