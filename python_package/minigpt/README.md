# minigpt

> 执行代码
```bash
# 1. 创建词表
python ./minigpt/build_vocab.py --data ./data/data.jsonl --output ./data/vocab.json
# 2. 拆分数据集
python ./minigpt/split_data.py --input ./data/data.jsonl --train_ratio 0.8
# 3. 训练模型
python ./minigpt/train.py
```
## 1. 构建词表

### 1.1 `build_vocab.py` — 词表构建工具

这个示例中，先按照中文字符生成词表，具体如下：
- **分词单位**：每一个**单字**（包括汉字、标点、数字、英文字母等）作为一个 token。
- **示例**：
  - 文本：`"你好吗？"`
  - 分词结果：`["你", "好", "吗", "？"]`
  - 每个字对应一个 ID

- **特殊 Token**：
  - `<pad>`：填充
  - `<unk>`：未登录字（理论上不会出现，因为你用全训练集构建词表）
  - `<sep>`：分隔符（用于分隔 question 和 answer）


用于从问答（QA）数据集中提取所有字符，并生成模型训练所需的词表文件 `vocab.json`。该词表将被 `Tokenizer` 类加载，用于将文本转换为模型可处理的 token ID 序列。

**输入数据格式**
脚本要求输入文件为 **JSONL（JSON Lines）格式**，即每行包含一个独立的 JSON 对象，且必须包含 `question` 和 `answer` 字段。

**示例 (`data/data.json`)**：
```json
{"question": "你好，最近怎么样？", "answer": "我很好，谢谢！"}
{"question": "你喜欢旅行吗？", "answer": "是的，我非常喜欢。"}
```
> ⚠️ 注意：不是整个文件是一个 JSON 数组，而是**每行一个 JSON 对象**。

```bash
python build_vocab.py --data <训练数据路径> [--output <输出路径>]
```

```bash
python ./minigpt/build_vocab.py --data ./data/data.jsonl --output ./data/vocab.json
```

- **特殊 token 固定包含**：`<pad>`（填充）、`<unk>`（未知字符）、`<sep>`（分隔符）
- 所有中文字符、标点、数字、字母等均按 Unicode 排序后分配 ID
- 词表大小 = 3（特殊 token） + 唯一字符数

> **注意**：如果更新了数据集，需要重新运行此脚本以更新词表。

```python
import enum
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

## 2. tokenizer

在构建词表后，创建对应的 Tokenizer 类，用于将文本转换为模型可处理的 token ID 序列。

```python
import json
import token


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

测试一下：
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
    """"

```

## 3. 拆分数据集
我们将数据集拆分为训练集和验证集，并保存为 JSONL 文件。

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

    print(f"✅ Split completed!")
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

```bash
python ./minigpt/split_data.py --input ./data/data.jsonl --train_ratio 0.8
```

## 4. 构建Dataset
构建一个自定义的 PyTorch Dataset 类，用于加载数据集并生成输入序列和标签。

```python
import torch
from torch.utils.data import Dataset, DataLoader
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
```

测试一下
```python
if __name__ == "__main__":
    tokenizer = Tokenizer("./data/vocab.json")
    dataset = QADataset("./data/train.jsonl", tokenizer, max_length=128)

    print(f"数据集大小：{len(dataset)}")

    item = dataset[0]
    print(item)
    print(tokenizer.decode(item["input_ids"].tolist()))
    print(tokenizer.decode(item["targets"].tolist()))
"""
枣树生长的产物分类为何类？<sep>枣树生长的产物属于果实类。<sep>
树生长的产物分类为何类？<sep>枣树生长的产物属于果实类。<sep>
"""
```

## 5. 先把训练逻辑写好

TODO: 添加代码

## 6. GPTLMHeadModel

这里采用的是 decoder-only 的 GPT 模型，即只包含解码器部分，不包含编码器部分。

整体思路，经过Transformer模型后的 hidden_state 作为输入，给到一个nn.Linear 层，得到最终的输出。

```python
from dataclasses import dataclass
import torch.nn as nn
import torch
from typing import Optional
import torch.nn.functional as F


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
    block_size: int = 128  # 最大上下文长度（位置编码最大支持长度）
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
        self.resid_dropput = nn.Dropout(config.dropout)

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

## 7. 模型训练
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
    batch_size = 32
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

```bash
python ./minigpt/train.py
```

## 8. 生成回答


## X. Qwen3
Qwen3的代码结构学习，详见 minigpt/qwen3 代码。
