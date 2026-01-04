# -*- coding: utf-8 -*-
"""
手写 Qwen3-0.6B 模型推理代码（无 KV Cache）
适用于学习目的，但在低显存设备（如 4GB GPU）上需谨慎使用。
"""

import torch
from safetensors.torch import load_file  # 安全加载 .safetensors 权重文件
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer  # 使用 Hugging Face 的 fast tokenizer


# ==============================================================================
# 1. RoPE（旋转位置编码）实现
# Qwen3 使用 RoPE 对 query 和 key 进行位置感知编码
# ==============================================================================
def apply_rotary_pos_emb(q, k, position_ids, head_dim, rope_theta=1000000.0):
    """
    对 query 和 key 应用旋转位置编码（RoPE）

    Args:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_key_value_heads, seq_len, head_dim]
        position_ids: [batch_size, seq_len]，每个 token 的位置索引
        head_dim: 每个注意力头的维度（Qwen3 中为 128）
        rope_theta: RoPE 的基底频率参数（Qwen3 使用 1e6）

    Returns:
        q_embed, k_embed: 经过 RoPE 编码后的 query 和 key
    """
    device = q.device
    # 计算频率反比：inv_freq = 1 / (theta^(i/d))，i 为偶数索引
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # freqs = position_ids * inv_freq → [batch_size, seq_len, head_dim//2]
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)

    # 将频率扩展为完整维度（实部+虚部）
    emb = torch.cat([freqs, freqs], dim=-1)  # [batch_size, seq_len, head_dim]

    # 计算 cos 和 sin，并扩展维度以匹配 q/k 的形状 [batch, 1, seq_len, head_dim]
    cos = emb.cos().unsqueeze(1).to(q.dtype)
    sin = emb.sin().unsqueeze(1).to(q.dtype)

    def rotate_half(x):
        """将向量后半部分移到前面并取负，实现复数乘法的旋转效果"""
        x1, x2 = x.chunk(2, dim=-1)  # 拆分为两半
        return torch.cat((-x2, x1), dim=-1)

    # 应用 RoPE: x * cos + rotate_half(x) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# 2. RMSNorm 实现（Qwen3 使用的 LayerNorm 变体）
# 不使用可学习的 bias，仅缩放
# ==============================================================================
class SelfQwen3RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可学习缩放因子
        self.variance_epsilon = eps  # 防止除零的小常数

    def forward(self, hidden_states):
        """
        RMSNorm 公式: x * weight / sqrt(mean(x^2) + eps)
        注意：先转为 float32 计算以避免精度损失，再转回原 dtype
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)  # 在最后一个维度求均值
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ==============================================================================
# 3. 多头注意力模块（支持 GQA：Grouped Query Attention）
# Qwen3-0.6B: num_heads=16, num_key_value_heads=8 → 每 2 个 Q 共享 1 个 K/V
# ==============================================================================
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 1024
        self.num_heads = config.num_attention_heads  # 16
        self.num_key_value_heads = config.num_key_value_heads  # 8
        self.head_dim = (
            config.head_dim
        )  # 128（注意：hidden_size ≠ num_heads * head_dim！）
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads  # 2

        # 投影层：注意输出维度基于 head_dim 而非 hidden_size
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # Qwen3 特有：在 RoPE 前对 Q/K 做 RMSNorm
        self.q_norm = SelfQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = SelfQwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rope_theta = config.rope_theta

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        bsz, q_len, _ = hidden_states.size()

        # 投影并 reshape 为多头格式 [batch, num_heads, seq_len, head_dim]
        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Qwen3 特有：先对 Q/K 做 RMSNorm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # 如果未提供 position_ids，则使用默认递增序列
        if position_ids is None:
            position_ids = torch.arange(q_len, device=hidden_states.device).unsqueeze(0)

        # 应用 RoPE 位置编码
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, position_ids, self.head_dim, self.rope_theta
        )

        # GQA：将 K/V 扩展以匹配 Q 的头数（8 heads → 16 heads）
        if self.num_key_value_groups > 1:
            # 扩展维度后 flatten 合并
            key_states = (
                key_states.unsqueeze(2)
                .expand(-1, -1, self.num_key_value_groups, -1, -1)
                .flatten(1, 2)
            )
            value_states = (
                value_states.unsqueeze(2)
                .expand(-1, -1, self.num_key_value_groups, -1, -1)
                .flatten(1, 2)
            )

        # 计算缩放点积注意力
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (
            self.head_dim**0.5
        )

        # 应用注意力掩码（如因果掩码）
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # softmax 归一化（注意：先转 float32 提高数值稳定性）
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )
        attn_output = torch.matmul(attn_weights, value_states)

        # 合并注意力头并投影回 hidden_size
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


# ==============================================================================
# 4. MLP 模块（SwiGLU 激活）
# Qwen3 使用 gate_proj + up_proj + silu 激活
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        # SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==============================================================================
# 5. 单个 Decoder 层（Attention + MLP + 残差连接）
# ==============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = SelfQwen3RMSNorm(config.hidden_size)  # Pre-norm 结构
        self.post_attention_layernorm = SelfQwen3RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden_states, position_ids=None, attention_mask=None):
        # 第一个残差块：Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        # 第二个残差块：MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# ==============================================================================
# 6. 完整语言模型（Qwen3ForCausalLM）
# ==============================================================================
class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        print("Initializing Qwen3 model...")
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = SelfQwen3RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 词嵌入与输出层权重共享（tie embeddings）
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        bsz, q_len = input_ids.shape

        # 自动生成位置 ID（从 0 到 seq_len-1）
        position_ids = torch.arange(
            q_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        # 构建因果注意力掩码（下三角为 0，上三角为 -inf）
        causal_mask = torch.triu(
            torch.full(
                (q_len, q_len),
                float("-inf"),
                dtype=torch.float32,
                device=input_ids.device,
            ),
            diagonal=1,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # 嵌入输入 tokens
        hidden_states = self.embed_tokens(input_ids)

        # 逐层前向传播
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_ids=position_ids, attention_mask=causal_mask
            )

        # 最终归一化 + 语言模型头
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits


# ==============================================================================
# 7. Qwen3-0.6B 模型配置（硬编码）
# 来自官方 config.json
# ==============================================================================
class SelfQwen3Config:
    architectures = ["Qwen3ForCausalLM"]
    attention_bias = False
    attention_dropout = 0.0
    bos_token_id = 151643  # Begin-of-Sequence token
    eos_token_id = 151645  # End-of-Sequence token
    head_dim = 128  # 每个注意力头的维度
    hidden_act = "silu"
    hidden_size = 1024  # 隐藏层维度
    initializer_range = 0.02
    intermediate_size = 3072  # MLP 中间层大小
    max_position_embeddings = 40960
    num_attention_heads = 16
    num_hidden_layers = 28  # 总共 28 层
    num_key_value_heads = 8  # GQA 设置
    rms_norm_eps = 1e-06
    rope_theta = 1000000  # RoPE 基频
    tie_word_embeddings = True
    vocab_size = 151936  # 词表大小（包含特殊 token）


# ==============================================================================
# 8. 主函数：加载模型 + 生成文本
# ⚠️ 注意：此实现无 KV Cache，效率低，显存占用高！
# ==============================================================================
def main():
    # 初始化模型（CPU 上）
    config = SelfQwen3Config()
    model = Qwen3ForCausalLM(config)

    # 从 safetensors 文件加载权重
    state_dict = load_file("model/model.safetensors")

    # 移除权重字典中的 "model." 前缀（适配 Hugging Face 格式）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_state_dict[k[len("model.") :]] = v  # 去掉 "model."
        else:
            new_state_dict[k] = v

    # 加载权重（严格匹配）
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    print("✅ Model loaded successfully!")

    # 加载分词器
    tokenizer = Tokenizer.from_file("model/tokenizer.json")

    # 构造聊天模板（⚠️ 此处未使用官方 apply_chat_template，可能格式不标准）
    message = "<|im_start|>user明天做点啥<|im_end|>><|im_start|>assistant"
    input_ids = tokenizer.encode(message).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    print(f"Input token IDs shape: {input_ids.shape}")

    # 🔥 自回归生成（无 KV Cache！每次重新计算全部历史）
    with torch.no_grad():
        for step in range(1000):  # 最多生成 1000 个 token（极易 OOM！）
            # 前向推理（⚠️ 整个序列重新计算！）
            logits = model(input_ids)

            # 取最后一个 token 的 logits 并 greedy 采样
            next_token_logits = logits[:, -1, :] / 1.0
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 遇到 EOS token 则停止
            if next_token.item() == config.eos_token_id:
                break

            # 拼接新 token
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # 解码输出（跳过特殊 token）
        output_text = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        print("\n🤖 Generated Output:")
        print(output_text)


if __name__ == "__main__":
    main()
