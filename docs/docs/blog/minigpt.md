---
title: ⚡MiniGPT —— 基于代码理解 transformer
date: 2026-01-05
tags: [LLMs, Pytorch]
description: 使用 Pytorch 从零构建词表、构建 decoder-only 的 类GPT-2 模型，从 0 到 1 实现一个自定义的对话模型。模型整体 Transformer Only Decoder 作为核心架构，由多个相同的层堆叠而成，每个层包括自注意力机制、位置编码和前馈神经网络。
draft: false
---

# ⚡MiniGPT —— 基于代码理解 transformer

> 这是一篇记录自己理解 transformer 模型的笔记，主要是使用Pytorch实现一个基础的 GPT-2 模型。

## 1. 数据集
使用对话-百科（中文）数据集，涵盖了美食、城市、企业家、汽车、明星八卦、生活常识、日常对话 等信息。数据集下载地址：[here](https://modelscope.cn/datasets/qiaojiedongfeng/qiaojiedongfeng/summary)

> 这个数据集在 minigpt 的代码中已经包含，可以直接使用。

数据格式如下：
```console
{"question": "你好，最近怎么样？", "answer": "你好！我最近还不错，谢谢。"}
{"question": "今天天气如何？", "answer": "今天的天气很晴朗。"}
{"question": "你喜欢旅行吗？", "answer": "是的，我非常喜欢旅行。"}
{"question": "你最喜欢的食物是什么？", "answer": "我最喜欢的食物是寿司。"}
{"question": "你有什么兴趣爱好？", "answer": "我喜欢阅读和运动。"}
{"question": "你最喜欢的电影是什么？", "answer": "我最喜欢的电影是《肖申克的救赎》。"}
```

## 2. 构建词表
因为数据集是中文，所以这里用一个字作为一个词，并在这个基础上把标点符号以及表情符号都纳入词表中。同时，词表中还需要三个特殊的词： `<pad>`用于表示占位、`<unk>` 用于表示未知、 `<sep>`表示分隔符，用于分隔 question 和 answer。

- **示例**：
  - 文本：`"你好吗？"`
  - 分词结果：`["你", "好", "吗", "？"]`
  - 每个字对应一个 ID

```python
import json
import argparse
from collections import Counter


def build_vocab(data_path: str, output_path: str):
    counter = Counter()

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                # 收集 question 和 answer 中的所有字符
                counter.update(item["question"])
                counter.update(item["answer"])
            except Exception as e:
                print(f"Warning: skip invalid line: {line[:50]}... | Error: {e}")

    # 获取所有唯一字符
    chars = sorted(counter.keys())

    # 添加特殊 token
    special_tokens = ["<pad>", "<unk>", "<sep>"]

    # 构建 word2id：先放特殊 token，再放字符（顺序固定便于复现）
    word2id = {token: i for i, token in enumerate(special_tokens)}
    for char in chars:
        if char not in special_tokens:  # 防御：跳过特殊 token
            word2id[char] = len(word2id)  # 自动递增

    # 构建 id2word
    id2word = {i: token for token, i in word2id.items()}

    vocab = {"word2id": word2id, "id2word": id2word}

    # 保存 vocab
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"✅ Vocabulary built and saved to {output_path}")
    print(f"   Total tokens: {len(word2id)}")
    print(f"   Special tokens: {special_tokens}")
    print(f"   Sample chars: {chars[:10]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vocabulary from QA dataset.")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data (JSONL format)"
    )
    parser.add_argument(
        "--output", type=str, default="vocab.json", help="Output vocab file path"
    )
    args = parser.parse_args()

    build_vocab(args.data, args.output)
```
执行命令后将词表保存在data目录下，生成的词一共4825个，该词表将被 `Tokenizer` 类加载，用于将文本转换为模型可处理的 token ID 序列。

```bash
python ./minigpt/build_vocab.py --data ./data/data.jsonl --output ./data/vocab.json
```

## 3. 创建 Tokenizer 类
Tokenizer 类用于将输入的文本数据进行分词，并生成对应的索引序列，同时将模型的输出转换为可读文本。

```python
import json


class Tokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        self.word2id = vocab["word2id"]
        self.id2word = {int(k): v for k, v in vocab["id2word"].items()}

        # 固定特殊 token ID
        self.pad_token_id = self.word2id["<pad>"]
        self.unk_token_id = self.word2id["<unk>"]
        self.sep_token_id = self.word2id["<sep>"]

    def encode(
        self,
        question: str,
        answer: str,
        max_length: int = 128,
        pad_to_max_length: bool = True,
    ):
        """将问答对编码为 token ID 序列。"""
        tokens = []

        # encode question
        for char in question:
            tokens.append(self.word2id.get(char, self.unk_token_id))
        tokens.append(self.sep_token_id)  # 添加分隔符

        # encode answer
        if answer is not None:
            for char in answer:
                tokens.append(self.word2id.get(char, self.unk_token_id))

            tokens.append(self.sep_token_id)

        # 构建 attention mask（1=真实 token，0=padding）
        attn_mask = [1] * len(tokens)

        # 截断或填充
        if pad_to_max_length:
            if len(tokens) > max_length:
                # 截断（保留开头）
                tokens = tokens[:max_length]
                attn_mask = attn_mask[:max_length]
            else:
                # 填充
                pad_len = max_length - len(tokens)
                tokens.extend([self.pad_token_id] * pad_len)
                attn_mask.extend([0] * pad_len)

        return tokens, attn_mask

    def decode(self, ids):
        """将 token ID 列表解码为原始文本（跳过 <pad>）。"""
        return "".join(
            self.id2word[i] for i in ids if i != self.pad_token_id  # 跳过填充符
        )

    def get_vocab_size(self):
        return len(self.id2word)
```

使用一个简单的例子来测试这个类：
```python
if __name__ == "__main__":
    question = "你好，最近怎么样？"
    answer = "我很好，谢谢！"

    tokenizer = Tokenizer("./data/vocab.json")

    input_ids, attn_mask = tokenizer.encode(question, answer, max_length=32)
    print(input_ids)
    print(tokenizer.decode(input_ids))
    """------ result ------
    [368, 1086, 4810, 2005, 4169, 1521, 240, 2103, 4816, 2, 1646, 1480, 1086, 4810, 3989, 3989, 4806, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    你好，最近怎么样？<sep>我很好，谢谢！<sep>
    """
```

## 4. 划分数据
将数据打乱后进行划分，这里我划分成80%训练集和20%验证集。
```python
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
```

执行下面的命令拆分数据集，如果是在没有GPU资源或GPU显存不足的情况下，可以调整代码，取一小部分数据进行测试。
```bash
python ./minigpt/split_data.py --input ./data/data.jsonl --train_ratio 0.8
```

## 5. 查看训练数据的token统计

执行如下代码，查看训练数据的token统计信息。
```bash
python ./minigpt/dataset_stats.py --train_data ./data/train.jsonl --vocab_path ./data/vocab.json
```

可以看到95%的token长度都在140以内，因此后面我们在后面训练的时候，可以将最大的输入长度设置为140。

<Image 
src='assets/minigpt_dataset_stats.png'
card=true
/>

## 6. 创建 Dataset
这里没有考虑太多优化的问题，仅做学习，因此创建的QADataset类比较简单。

```python
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
    print(tokenizer.decode(item["input_ids"].tolist()))
    print(tokenizer.decode(item["targets"].tolist()))
```

我们看一下 idx 为 0 的样本中数据经过 decode 后的结果：

```console
枣树生长的产物分类为何类？<sep>枣树生长的产物属于果实类。<sep>
树生长的产物分类为何类？<sep>枣树生长的产物属于果实类。<sep>
```

## 7. 模型搭建
### 7.1 注意力机制背景

首先用一个最简单的例子说明注意力层

- **Query**: 通过外卖App点外卖，个人的关注点是饭店口味是否麻辣和价格，例如 Query = [0.9, 0.8] 表示我希望饭店的‘麻辣程度’和‘便宜程度’都尽可能高。**为此，Key 的两个维度必须分别定义为‘越辣值越大’和‘越便宜值越大’。”** 简单理解，Query就是加权平均的权重。

- **Key**: 假设有三家饭店，每家饭店使用菜品麻辣程度和价格作为其饭店特征，例如 Key = [0.8, 0.7], 第一个维度表示饭店菜品的麻辣程度的得分（越辣得分越高），第二个维度表示饭店价格的便宜程度（值越接近1，表示饭店越便宜），这个要与Query 的理解一致。

> 💡 用几个不恰当的例子（有个人主观因素😄）：
> - 粤菜馆：Key = [0.1, 0.3]   不辣，价格较高 
> - 川菜馆：Key = [0.9, 0.5]   麻辣，价格中等
> - 日式餐厅：Key = [0.3, 0.1] 不辣，价格太贵

- **Value**: 每个饭店能提供的有用信息，例如 Value = [0.9, 0.8] 可以表示用户综合评分0.9，出餐速度0.8。

> 💡 Value 不需要与 Query 和 Key 的维度数量和含义保持一致，Value可以是任何有用的信息。

**第一步：计算一下原始匹配分数**：

$$
\text{score}_i = \mathbf{q}^\top \mathbf{k}_i = 0.9 \cdot k_{i1} + 0.8 \cdot k_{i2}
$$

逐个计算：

| 饭店      | 计算过程                                        | 原始分数 |
| --------- | ----------------------------------------------- | -------- |
| A（粤菜） | $0.9 \times 0.1 + 0.8 \times 0.3 = 0.09 + 0.24$ | **0.33** |
| B（川菜） | $0.9 \times 0.9 + 0.8 \times 0.5 = 0.81 + 0.40$ | **1.21** |
| C（日料） | $0.9 \times 0.3 + 0.8 \times 0.1 = 0.27 + 0.08$ | **0.35** |

> 💡 川菜馆遥遥领先——又辣又相对便宜！

**第二步：Softmax 归一化 → 注意力得分（权重）**

$$
\alpha_i = \frac{\exp(\text{score}_i)}{\exp(0.33) + \exp(1.21) + \exp(0.35)}
$$

先算指数（使用计算器，保留4位小数）：

$$
\begin{aligned}
\exp(0.33) &\approx 1.3910 \\
\exp(1.21) &\approx 3.3535 \\
\exp(0.35) &\approx 1.4191 \\
\text{总和} &= 1.3910 + 3.3535 + 1.4191 = 6.1636
\end{aligned}
$$

再计算每个饭店的**注意力得分（Attention Score / Weight）**：

| 饭店      | 公式          | 注意力得分（αᵢ） | 百分比    |
| --------- | ------------- | ---------------- | --------- |
| A（粤菜） | 1.3910/6.1636 | **0.2257**       | **22.6%** |
| B（川菜） | 3.3535/6.1636 | **0.5441**       | **54.4%** |
| C（日料） | 1.4191/6.1636 | **0.2302**       | **23.0%** |


- **川菜馆获得超过一半（54.4%）的注意力**  
  → 因为它最符合你“又辣又便宜”的偏好
- **粤菜馆和日料馆各占约 1/4**  
  → 虽然都不辣，但粤菜稍便宜（便宜程度 0.3 > 0.1），所以略高于日料

> 这些注意力得分决定了后续聚合 Value 时的**话语权**：  
> 川菜馆的评分和出餐速度对最终结果影响最大，之后将注意力得分乘以Value并求和后，得到的是一个上下文感知的推荐摘要 —— 不是某一家饭店，而是根据你的偏好动态融合三家饭店的综合表示，可以理解为是一个虚拟的饭店，用于指导后续的决策。

接下来思考，如果我的考虑维度很多呢？比如我既关注价格，又关注麻辣口味，还关注健康等等，是否可以将这些偏好维度都放进Query中？

理论上可以，但是实践上的效果可能不太好，因为维度太多，可能导致不同维度的相互干扰，所有的维度混在一起计算总的分数，导致最终的得分可能不是最合适的。

多头的目的在这里就体现出来了，我可以设置多个头，每个头都关注几个维度，然后把多个头的结果进行拼接，得到一个更适合的表示。

进一步解释（推理阶段的）KV缓存，因为个人的偏好 Query 可能会变，但是饭店的特征(Key/Value) 不会变，所以只要饭店没有换菜单，没调价，它们的 Key 和 Value 都不会变，所以可以缓存起来，下次查询的时候直接从缓存中取，省得重复计算。（这个例子实际上可能不太恰当，但是勉强可以理解）

### 7.2 注意力机制实现

回到miniGPT的实现上，我们把注意力机制的公式写一下：
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这里需要注意，我们要在这个公式的基础上，增加注意力掩码，由于输入序列可能是不同的长度，但矩阵运算时需要固定的大小，因此针对长度不足的序列，使用padding作填充，但是这些padding的信息是没有意义的，因此需要将这些padding的位置的注意力掩码设置为0，这样在计算softmax的时候，这些位置的注意力分数就会变成0，从而被忽略。


```python
import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k  # 输入的维度，用于计算缩放因子

    def forward(self, q, k, v, attn_mask):
        """
        Args:
            q: [batch_size, n_heads, len_q, d_k]
            k: [batch_size, n_heads, len_k, d_k]
            v: [batch_size, n_heads, len_v, d_v]  (注意: len_k == len_v)
            attn_mask: [batch_size, n_heads, len_q, len_k] 或 None
        Returns:
            context: [batch_size, n_heads, len_q, d_v]
            attn:    [batch_size, n_heads, len_q, len_k]
        """
        # step 1: 计算 QK^T / sqrt(d_k) shape: [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)

        # step 2: 应用 attn_mask
        if attn_mask is not None:
            # masked_fill_: 将 mask=False 的位置替换为 -1e9
            # 这里如果替换为0，后面的softmax计算时候，如果其他项的值也比较小
            # 那么softmax之后的结果可能还有权重，所以设置一个很大的负数，可以避免这种情况
            scores = scores.masked_fill(~attn_mask, -1e9)

        # step 3: softmax 得到 attn
        attn = nn.Softmax(dim=-1)(scores)  # shape: [batch_size, n_heads, len_q, len_k]

        # step 4: attn * v
        context = torch.matmul(attn, v)  # shape: [batch_size, n_heads, len_q, head_dim]

        return context, attn


if __name__ == "__main__":
    # 用户偏好 Query
    q = torch.tensor([[[[0.9, 0.8]]]])  # shape: [1, 1, 1, 2]

    # 三家饭店的 Key: [麻辣程度, 便宜程度]
    k = torch.tensor(
        [[[[0.1, 0.3], [0.9, 0.5], [0.3, 0.1]]]]  # 粤菜馆  # 川菜馆  # 日式餐厅
    )  # shape: [1, 1, 3, 2]

    # 三家饭店的 Value: [评分, 出餐速度]
    v = torch.tensor(
        [[[[0.5, 0.9], [0.9, 0.6], [0.7, 0.8]]]]  # 粤菜馆  # 川菜馆  # 日式餐厅
    )  # shape: [1, 1, 3, 2]

    # 没有mask
    mask = None

    attention = ScaledDotProductAttention(d_k=2)

    context, attn_weights = attention(q, k, v, mask)

    print("=== Attention Weights (α) ===")
    print(attn_weights.squeeze())  # shape: [3]

    print("\n=== Output Context (Weighted Value) ===")
    print(context.squeeze())  # shape: [2]
```

进一步验证一下，我不喜欢日式餐厅，直接屏蔽 `mask[:,:,:,2] = True`

```python
mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
mask[:, :, :, 2] = False
```

### 7.3 多头注意力机制实现
在上面的基础上，我们实现多头注意力机制
```python {5-79}
import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # 线性变换层
        self.w_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_heads * d_v, bias=False)

        # 输出层
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        # 注意力模块
        self.attention = ScaledDotProductAttention(d_k)

        # LayerNorm层
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: [batch_size, len_q, d_model]
            k: [batch_size, len_k, d_model]
            v: [batch_size, len_v, d_model]  (len_k == len_v)
            attention_mask: [batch_size, len_q, len_k] or None
        Returns:
            output: [batch_size, len_q, d_model]
            attn:   [batch_size, n_heads, len_q, len_k]
        """
        residual = q  # 残差连接
        batch_size = q.size(0)

        # 将q、k、v进行线性映射拆分
        # [Batch_size, len_q, d_model] -> [Batch_size, len_q, n_heads, d_k]
        q_proj = self.w_q(q).view(
            batch_size, -1, self.n_heads, self.d_k
        )  # [batch_size, len_q, n_heads, d_k]
        k_proj = self.w_k(k).view(
            batch_size, -1, self.n_heads, self.d_k
        )  # [batch_size, len_k, n_heads, d_k]
        v_proj = self.w_v(v).view(
            batch_size, -1, self.n_heads, self.d_v
        )  # [batch_size, len_v, n_heads, d_v]

        # Transpose to [B, n_heads, Len_v, d_v]
        q_proj = q_proj.transpose(1, 2)
        k_proj = k_proj.transpose(1, 2)
        v_proj = v_proj.transpose(1, 2)

        if attn_mask is not None:  # 如果attn_mask不为None，则进行mask操作
            # [B, Lq, Lk] -> [B, 1, Lq, Lk] -> [B, n_heads, Lq, Lk]
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        else:
            attn_mask = None

        # scaled dot-product attention
        context, attn = self.attention(q_proj, k_proj, v_proj, attn_mask)

        # concat heads
        # [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )

        output = self.fc(context)

        output = self.layer_norm(output + residual)

        return output, attn


if __name__ == "__main__":
    # 模拟三家饭店的原始特征 [辣, 便宜, 健康, 评分]
    # shape: [batch=1, seq_len=3, d_model=4]
    k_v_input = torch.tensor(
        [
            [
                [0.1, 0.3, 0.8, 0.5],  # 粤菜馆
                [0.9, 0.5, 0.2, 0.9],  # 川菜馆
                [0.3, 0.1, 0.9, 0.7],  # 日式餐厅
            ]
        ]
    )

    # Query：当前用户偏好（也用同样4维表示）
    q_input = torch.tensor([[[0.9, 0.8, 0.3, 0.6]]])  # 我喜欢辣、便宜，不太在意健康

    # 测试1: 无 mask
    mha = MultiHeadAttention(n_heads=2, d_model=4, d_k=2, d_v=2)
    output, attn_weights = mha(q_input, k_v_input, k_v_input, attn_mask=None)
    print("=== No Mask ===")
    print("Output shape:", output.shape)  # [1, 1, 4]
    print("Attn shape:", attn_weights.shape)  # [1, 2, 1, 3]
    print("Head 1 weights:", attn_weights[0, 0, 0].tolist())
    print("Head 2 weights:", attn_weights[0, 1, 0].tolist())

    # 测试2: 屏蔽日式餐厅 (index=2)
    mask = torch.ones(1, 1, 3, dtype=torch.bool)
    mask[0, 0, 2] = False  # False = 屏蔽
    output2, attn_weights2 = mha(q_input, k_v_input, k_v_input, attn_mask=mask)
    print("\n=== With Mask ===")
    print("Head 1 weights:", attn_weights2[0, 0, 0].tolist())
    print("Head 2 weights:", attn_weights2[0, 1, 0].tolist())
```

### 7.4 改造成 CasualSelfAttention
我们抛弃掉自己创建的注意力机制，改用 PyTorch 提供的实现，同时把整体的代码适配 miniGPT 的模型上。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    模型超参数配置类（使用 dataclass 简洁定义）

    默认值参考 GPT-2 small，但 vocab_size 根据中文任务调整
    """

    n_embd: int = 768  # token embedding 维度（也即 hidden size）
    n_head: int = 8  # 多头注意力头数（必须整除 n_embd）
    n_layer: int = 6  # Transformer block 层数
    dropout: float = 0.1  # 所有 dropout 层的丢弃率
    block_size: int = 256  # 最大上下文长度（位置编码最大支持长度）
    vocab_size: int = 4825  # 词表大小（根据实际 tokenizer 决定


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0
        self.config = config
        # 将 embedding 分成了多个 head
        self.head_dim = config.n_embd // config.n_head
        # 创建 query、key、value 投影
        # 将 [B, T, C] 一次性投影为 [B, T, 3 * C]，然后再拆分为 Q/K/V
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 输出投影：将多头拼接后的向量映射回原始维度
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # 注意力输出的残差连接后加 dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]
            attention_mask: 可选，padding 掩码，形状 [batch_size, seq_len]
                - 1 表示有效 token
                - 0 表示 padding token

        Returns:
            输出张量，形状 [batch_size, seq_len, n_embd]
        """
        # batch, sequence length, embedding dim
        B, T, C = x.shape

        # === 步骤1: QKV 融合投影 ===
        qkv = self.qkv_proj(x)
        # 拆分成 Q, K, V, 维度均是 [B, T, C]
        q, k, v = qkv.chunk(3, dim=-1)

        # === 步骤2: 多头分割与转置 ===
        # 将每个头的维度从C拆为 n_head, head_dim
        # 然后转置为 [B, n_head, T, head_dim] 以便进行批量矩阵乘
        q = q.view(B, T, self.config.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_head, self.head_dim).transpose(1, 2)

        # === 步骤3: 构建复合注意力掩码 ===

        attn_mask = None
        if attention_mask is not None:
            # padding 掩码: [B, T] -> bool
            key_padding_mask = attention_mask.to(torch.bool)

            # 因果掩码：下三角矩阵[T, T]，确保只能 attend 到当前及左侧
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device)
            )

            # 拓展维度以支持广播
            # key_padding_mask: [B, 1, 1, T]
            # causal_mask: [1, 1, T, T]
            key_padding_mask = key_padding_mask[:, None, None, :]
            causal_mask = causal_mask[None, None, :, :]

            # 逻辑与：只有同时满足“非 padding”和“在左侧”才为 True
            attn_mask = causal_mask & key_padding_mask  # [B, 1, T, T]

        # === 步骤4: 高效注意力计算 ===
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.config.dropout if self.training else 0,
            is_causal=False,  # NOTE：我们已手动构建因果掩码，故设为 False
        )

        # 输出 y: [B, n_head, T, head_dim]

        # === 步骤5: 合并多头并投影 ===
        # 转置回 [B, T, n_head, head_dim] -> 合并最后两维 -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 经过输出投影 + 残差 dropout
        y = self.resid_dropout(self.out_proj(y))
        return y


# ========================
# 测试代码
# ========================
if __name__ == "__main__":
    torch.manual_seed(42)

    # 配置
    config = GPTConfig(n_embd=8, n_head=2, dropout=0.0)

    # 创建模块
    attn = CausalSelfAttention(config)

    # 输入：batch=2, seq_len=5, emb=8
    x = torch.randn(2, 5, 8)

    # attention_mask: 第一个样本有效长度=3，第二个=4
    attention_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]
    )  # 1=有效, 0=padding

    # 前向传播
    output = attn(x, attention_mask=attention_mask)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output (first sample, first token):", output[0, 0].detach().numpy())

    # 验证因果性：检查是否只 attend 到左侧
    # 手动计算 attention weights（仅用于验证，非必需）
    with torch.no_grad():
        qkv = attn.qkv_proj(x)
        q, k, _ = qkv.chunk(3, dim=-1)
        q = q.view(2, 5, 2, 4).transpose(1, 2)
        k = k.view(2, 5, 2, 4).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (4**0.5)  # [2, 2, 5, 5]
        # 应用因果掩码（下三角）
        causal_mask = torch.tril(torch.ones(5, 5, dtype=torch.bool))
        scores_masked = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = F.softmax(scores_masked, dim=-1)
        print("\nAttention weights for head 0, sample 0:")
        print(attn_weights[0, 0].numpy())
        # 应该是下三角，且每行和为1
```

### 7.5 构建完整的GPT模型
剩下的模型结构相对简单，这里就不专门介绍了，我们直接构建整体的GPT模型。

**前馈神经网络MLP层**
```python
class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network, FFN）

    GPT 标准结构：
      Linear(n_embd, 4*n_embd) → GELU → Linear(4*n_embd, n_embd)

    为什么是 4 倍？—— GPT 论文发现此比例在性能与计算间取得良好平衡
    """

    def __init__(self, config: GPTConfig):
        super().__init__()

        # 升维：扩大表示能力
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # 降维：回到原始维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        # 输出 dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)  # GELU 激活函数（GPT 系列标准选择）
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

**Transformer 解码器块**
```python
class Block(nn.Module):
    """
    Transformer 解码器块（Decoder Block）

    结构（Pre-LN 风格）：
      x → LayerNorm → Attention → Add → LayerNorm → MLP → Add → Output

    Pre-LN 优势：
      - 训练更稳定（梯度不会随层数爆炸）
      - 无需学习率 warmup（在中小模型中效果显著）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # 第一个 LayerNorm（用于 Attention 前）
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        # 第二个 LayerNorm（用于 MLP 前）
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # 注意力子层（带残差连接）
        x = x + self.attn(self.ln_1(x), attention_mask)
        # MLP 子层（带残差连接）
        x = x + self.mlp(self.ln_2(x))
        return x
```

**Transformer 结构**
```python
class Transformer(nn.Module):
    """
    包含：
      - Token Embedding
      - 可学习位置编码（Learned Positional Embedding）
      - N 个 Block
      - 最终 LayerNorm（Pre-LN 结构的一部分）
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token Embedding: [vocab_size, n_embd]
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # 位置编码: [block_size, n_embd]（可学习，非正弦）
        # 注意：GPT-2 使用可学习位置编码，而非原始 Transformer 的固定编码
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # 输入 dropout
        self.drop = nn.Dropout(config.dropout)

        # 堆叠 N 个 Transformer Block
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # 最终 LayerNorm（Pre-LN 结构要求在最后加一次 LN）
        self.ln_f = nn.LayerNorm(config.n_embd)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T]，token ID 序列
            attention_mask: [B, T]，可选 padding 掩码

        Returns:
            hidden_states: [B, T, n_embd]
        """
        B, T = input_ids.shape
        # 安全检查：序列长度不能超过 block_size
        assert (
            T <= self.config.block_size
        ), f"序列长度 {T} 超出最大上下文长度 {self.config.block_size}"

        # === 获取 token embedding ===
        tok_emb = self.wte(input_ids)  # [B, T, n_embd]

        # === 获取位置 embedding ===
        # 使用 arange 生成位置索引 [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pos_emb = self.wpe(pos)  # [T, n_embd]

        # === 合并 token + 位置 embedding ===
        x = self.drop(tok_emb + pos_emb)  # [B, T, n_embd]

        # === 通过所有 Transformer Block ===
        for block in self.blocks:
            x = block(x, attention_mask)

        # === 最终 LayerNorm ===
        x = self.ln_f(x)
        return x
```

**完整的GPT模型**
```python
class GPTLMHeadModel(nn.Module):
    """
    完整的 GPT 语言模型（含语言建模头）

    关键特性：
      - 权重绑定（Weight Tying）：wte.weight = lm_head.weight
        * 减少参数量
        * 提升训练稳定性（Press & Wolf, 2017）
      - 自回归语言建模：预测下一个 token
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)

        # 语言建模头：将 hidden state 映射到 vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 🔑 权重绑定：共享 embedding 与 lm_head 的权重矩阵
        # 注意：必须在初始化后立即绑定，且两者 bias 均为 False
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化所有参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        参数初始化策略（遵循 GPT-2）
        - Linear / Embedding: Normal(0, 0.02)
        - Bias: 全零（但本模型无 bias）
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：输入 token IDs，输出每个位置的词汇表 logits

        Args:
            input_ids: [B, T]，输入 token ID 序列
            attention_mask: [B, T]，可选，用于屏蔽 padding

        Returns:
            logits: [B, T, vocab_size]，每个位置对整个词表的预测分数
        """
        # 通过 Transformer 编码器获取隐藏状态
        hidden_states = self.transformer(input_ids, attention_mask)

        # 通过语言建模头得到 logits
        logits = self.lm_head(hidden_states)  # [B, T, vocab_size]

        return logits
```

构建好后，我们看一下模型的整体情况：
```python
GPTLMHeadModel(
  (transformer): Transformer(
    (wte): Embedding(4825, 768)
    (wpe): Embedding(256, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (blocks): ModuleList(
      (0-5): 6 x Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): CausalSelfAttention(
          (qkv_proj): Linear(in_features=768, out_features=2304, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=False)
          (c_proj): Linear(in_features=3072, out_features=768, bias=False)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=4825, bias=False)
)
```
整体的参数量是 ~46.4M，相比于 GPT-2（124M） 还是小很多。

## 8. GPT 模型训练
这部分仅做基础训练，因此没有进行大量优化。

```python
import torch
from minigpt.qa_dataset import QADataset
from minigpt.tokenizer import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
from minigpt.model import GPTLMHeadModel, GPTConfig
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    progress_bar = tqdm(
        enumerate(train_loader),
        total=num_batches,
        desc="  Train",
        leave=False,
        unit="batch",
    )

    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    progress_bar = tqdm(
        enumerate(val_loader),
        total=num_batches,
        desc="  Val",
        leave=False,
        unit="batch",
    )

    for batch_idx, batch in progress_bar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    model_output_dir,
    writer,
):
    os.makedirs(model_output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")

        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, criterion, device)
        print(f"  Val Loss:   {val_loss:.4f}")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(model_output_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_path)
            print(f"  🎉 New best model saved (val loss: {val_loss:.4f})")

    print(f"\n✅ Training finished. Best validation loss: {best_val_loss:.4f}")


def main():
    # 配置路径
    train_path = "./data/train.jsonl"
    val_path = "./data/val.jsonl"
    vocab_path = "./data/vocab.json"

    # 超参数
    max_length = 128
    batch_size = 64
    lr = 1e-4
    epochs = 15

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 加载 tokenizer 和模型
    tokenizer = Tokenizer(vocab_path)
    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPTLMHeadModel(config).to(device)

    # 数据集
    train_dataset = QADataset(train_path, tokenizer, max_length)
    val_dataset = QADataset(val_path, tokenizer, max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # 优化器与损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # TensorBoard 日志
    writer = SummaryWriter("runs/minigpt")

    # 开始训练
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=epochs,
        model_output_dir="output",
        writer=writer,
    )

    writer.close()
    print("\n🎉 Training pipeline completed.")


if __name__ == "__main__":
    main()
```

> 💡 在上面的batch_size

## 9. 模型推理
### 9.1 为GPT添加generate方法
为了更加方便使用，且符合 Huggingface Transformer 库的标准范式，这里将 `generate` 方法补充添加到 `GPTLMHeadModel` 模型中。

```python
def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
    stop_token_ids: Optional[Union[int, List[int]]] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    do_sample: bool = False,
) -> torch.Tensor:
    """
    自回归文本生成方法（支持多种解码策略）

    Args:
        input_ids (torch.Tensor):
            初始输入 token IDs，形状 [batch_size, seq_len]
        max_new_tokens (int):
            最多生成的新 token 数量
        stop_token_ids (int or List[int], optional):
            遇到这些 token 时提前停止生成（如 <eos>）。每个样本独立判断是否停止。
        temperature (float):
            采样温度（>1 更随机，<1 更确定），仅在 do_sample=True 时生效
        top_k (int, optional):
            限制采样只在概率最高的 k 个 token 中进行
        do_sample (bool):
            是否使用随机采样（False 表示 greedy 解码）

    Returns:
        generated_ids (torch.Tensor):
            完整生成序列，形状 [batch_size, seq_len + actual_new_tokens]
            注意：不同样本可能生成不同长度，但返回张量是统一右填充（用最后一个有效 token 填充），
            若需严格截断，请在调用后按 stop token 手动处理。
    """
    self.eval()
    device = input_ids.device
    B, T = input_ids.shape

    # === 处理停止 token ===
    stop_tokens = set()
    if stop_token_ids is not None:
        if isinstance(stop_token_ids, int):
            stop_tokens.add(stop_token_ids)
        else:
            stop_tokens.update(stop_token_ids)

    # 转为 GPU tensor 用于向量化比较（避免 .item() 同步）
    stop_tensor = None
    if stop_tokens:
        stop_tensor = torch.tensor(list(stop_tokens), device=device, dtype=input_ids.dtype)  # [S]

    # === 初始化生成状态 ===
    generated = input_ids.clone()  # [B, T]
    finished = torch.zeros(B, dtype=torch.bool, device=device)  # [B]，记录每个样本是否已完成

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 提前终止：所有样本都已完成 或 超出上下文窗口
            if finished.all() or generated.size(1) >= self.config.block_size:
                break

            # 获取当前 logits（只取最后一个位置）
            logits = self(generated)  # [B, T_curr, vocab_size]
            next_token_logits = logits[:, -1, :]  # [B, vocab_size]

            # === 解码策略 ===
            if do_sample:
                # 温度缩放（确保 temperature > 0）
                if temperature <= 0:
                    raise ValueError("temperature must be > 0")
                next_token_logits = next_token_logits / temperature

                # Top-k 过滤
                if top_k is not None and top_k > 0:
                    k = min(top_k, next_token_logits.size(-1))
                    # 获取第 k 大的值作为阈值
                    values, _ = torch.topk(next_token_logits, k, dim=-1)
                    threshold = values[:, -1:]  # [B, 1]
                    # 将低于阈值的 logits 设为 -inf
                    next_token_logits = torch.where(
                        next_token_logits < threshold,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # [B, 1]

            # === 对已完成的样本，不更新 token（用原序列最后一个 token 占位）===
            # 注意：也可以用 pad_token，但模型未定义 pad_token_id，故复用 last token
            last_token = generated[:, -1:].clone()  # [B, 1]
            next_token = torch.where(finished.unsqueeze(1), last_token, next_token)

            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)  # [B, T+1]

            # === 更新 finished 状态（仅当设置了 stop_tokens 时）===
            if stop_tensor is not None:
                # 检查新生成的 token 是否在 stop_tokens 中 → [B]
                is_stop = (next_token == stop_tensor).any(dim=1)  # 广播比较 [B,1] vs [S] → [B,S] → any → [B]
                finished = finished | is_stop

    return generated
```

