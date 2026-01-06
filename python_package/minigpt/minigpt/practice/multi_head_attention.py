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
