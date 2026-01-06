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
            # masked_fill_: 将 mask=True 的位置替换为 -1e9
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
    # 屏蔽日式餐厅
    mask = torch.ones(1, 1, 1, 3, dtype=torch.bool)
    mask[0, 0, 0, 2] = False

    attention = ScaledDotProductAttention(d_k=2)

    context, attn_weights = attention(q, k, v, mask)

    print("=== Attention Weights (α) ===")
    print(attn_weights.squeeze())  # shape: [3]

    print("\n=== Output Context (Weighted Value) ===")
    print(context.squeeze())  # shape: [2]
