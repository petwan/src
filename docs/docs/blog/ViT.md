---
title: ⚡ViT
date: 2026-01-07
tags: [Pytorch]
description: 使用 Pytorch 从零构建词表、构建 decoder-only 的 类GPT-2 模型，从 0 到 1 实现一个自定义的对话模型。模型整体 Transformer Only Decoder 作为核心架构，由多个相同的层堆叠而成，每个层包括自注意力机制、位置编码和前馈神经网络。
draft: true
---

# ⚡ViT
<Image 
src="assets/ViT_arch.png" />

ViT将输入图片分为多个patch，然后将每个patch投影为固定长度的向量送入Transformer Encoder中，后续Encoder的操作和原始的Transformer一致。但因为是对图片进行分类，因此在输入序列中加入一个特殊的token，该token对应的输出即为最后的类别预测。

```bash
pip install torch torchvision matplotlib tqdm scipy numpy
```


