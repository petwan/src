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
