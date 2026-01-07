---
title: ⚡KNN学习与交互演示
date: 2026-01-07
tags: [Machine Learning]
description: KNN
draft: false
---

# ⚡KNN学习与交互演示

<script setup>
import KnnExplainer from '@src/knn/KnnExplainer.vue'
</script>


<KnnExplainer />


## 1. 概述

K近邻（K-Nearest Neighbors, KNN）是一种 **非参数化** 、**惰性学习**（lazy learning）的监督学习算法，广泛应用于分类和回归任务。其核心思想非常直观：<mark>**“物以类聚”**</mark> —— 一个样本的类别或数值由其在特征空间中最接近的K个邻居决定。

KNN 是一种基于实例的学习方法（instance-based learning），不显式构建模型，而是在预测阶段直接利用训练数据进行推理。

## 2. 算法原理

### 2.1 分类任务

对于一个待预测样本 $x$，KNN 的分类流程如下：

1. **计算距离**：计算 $x$ 与训练集中每个样本之间的距离（常用欧氏距离、曼哈顿距离等）。
2. **选取K个最近邻**：找出距离最小的K个训练样本。
3. **投票决策**：对这K个邻居的类别进行多数投票，得票最多的类别即为 $x$ 的预测类别。

数学表达：
$$
\hat{y} = \arg\max_{c \in C} \sum_{i=1}^{K} \mathbb{I}(y_i = c)
$$
其中，$C$ 为所有类别集合，$y_i$ 是第 $i$ 个最近邻的真实标签，$\mathbb{I}(\cdot)$ 为指示函数。

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# 示例数据
X = np.array([[1], [2], [3], [5], [8]])
y = np.array([2.1, 2.9, 4.0, 5.8, 8.1])

# 划分数据 & 标准化（重要！）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建 KNN 回归器（K=3，距离加权）
knn_reg = KNeighborsRegressor(n_neighbors=3, weights="distance")
knn_reg.fit(X_train_scaled, y_train)

# 预测
y_pred = knn_reg.predict(X_test_scaled)
print("预测值:", y_pred)
```

> 即使是一维特征，也做一下标准化；多维时**必须标准化**！


### 2.2 回归任务
**KNN Regression** 核心思想非常直观：**用最近的 K 个邻居的目标值的平均（或加权平均）作为预测值**。

给定一个待预测样本 $x_{\text{query}}$，KNN 回归的预测步骤如下：

1. **计算距离**：在训练集中计算 $x_{\text{query}}$ 与每个训练样本 $x_i$ 的距离（常用欧氏距离）。
2. **找出 K 个最近邻**：选择距离最小的 K 个训练样本。
3. **聚合目标值**：将这 K 个邻居对应的真实值 $y_i$ 进行平均（或加权平均），作为最终预测值 $\hat{y}$。


平均加权：

$$
\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i
$$

也可引入距离加权（如反距离加权）以提高精度：
$$
\hat{y} = \frac{\sum_{i=1}^{K} w_i y_i}{\sum_{i=1}^{K} w_i}
$$

其中，
$$ 
w_i = \frac{1}{d(x, x_i)+ \epsilon}
$$

$\epsilon$ 是一个很小的常数，用于避免分母为零。


### 2.3 简单总结

| 要素         | 说明                                                                                                                       |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **K 值选择** | - K 太小 → 过拟合、噪声敏感<br>- K 太大 → 欠拟合、边界模糊<br>- ✅ 通常选择奇数（避免平票）<br>- 可以通过交叉验证选取最优 K |
| **距离度量** | - 欧氏距离（默认）<br>- 曼哈顿距离、余弦相似度等<br>⚠️ 特征需**标准化/归一化**，否则量纲大的特征主导距离                    |
| **权重策略** | - uniform（等权）<br>- distance（距离倒数加权）<br>- 自定义核函数（如高斯核）                                              |
| **计算效率** | - 原始 KNN 是“惰性学习”，预测时才计算<br>- 大数据下可使用 KD-Tree、Ball Tree 或近似最近邻（ANN）加速                       |

## 3. 距离度量

KNN 的性能高度依赖于距离度量方式。常见选择包括：

- **欧氏距离（Euclidean Distance）**：
  $$
  d(x, x') = \sqrt{\sum_{j=1}^{n} (x_j - x'_j)^2}
  $$
- **曼哈顿距离（Manhattan Distance）**：
  $$
  d(x, x') = \sum_{j=1}^{n} |x_j - x'_j|
  $$
- **闵可夫斯基距离（Minkowski Distance）**（泛化形式）：
  $$
  d(x, x') = \left( \sum_{j=1}^{n} |x_j - x'_j|^p \right)^{1/p}
  $$
  当 $p=1$ 时为曼哈顿距离，$p=2$ 时为欧氏距离。

- **余弦相似度**（适用于高维稀疏数据，如文本）：
  $$
  \text{sim}(x, x') = \frac{x \cdot x'}{\|x\| \|x'\|}
  $$

> 注意：当特征尺度差异较大时，应进行 **标准化**（如 Z-score 或 Min-Max 缩放），否则距离会被大尺度特征主导。

## 4. K 值的选择
在KNN中，我们通过寻找k个最近邻的多数票（分类）或平均值（回归）来预测。当k很小时，决策边界会变得非常不规则，因为它只依赖于少数几个近邻点。这意味着决策边界会尽可能贴近训练数据，甚至对噪声和异常点也会敏感，从而导致复杂的模型。相反，当k较大时，决策边界会更平滑，因为每个点的预测受更多点的影响，模型变得简单。


K 是 KNN 中最关键的超参数：

- **K 过小**（如 K=1）：模型复杂，容易过拟合，对噪声敏感。
- **K 过大**：模型过于平滑，可能欠拟合，且计算开销大。
- **经验法则**：通常选择奇数（避免平票），并通过交叉验证（Cross-Validation）选取最优 K。

### 4.1 交叉验证

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris

# 加载数据
X, y = load_iris(return_X_y=True)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建 KNN 模型
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# 预测与评估
accuracy = knn.score(X_test_scaled, y_test)
print(f"Accuracy: {accuracy:.2f}")

# 交叉验证选择最优 K
scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_scaled, y_train, cv=5).mean()
    scores.append(score)

best_k = scores.index(max(scores)) + 1
print(f"Best K: {best_k}")
```
## 5. 优化与改进

为克服 KNN 的局限性，研究者提出了多种改进方法：

- **KD 树（k-d Tree）** 或 **Ball Tree**：用于加速近邻搜索（适用于低维空间）。
- **局部敏感哈希（LSH）**：适用于高维近似最近邻搜索。
- **加权 KNN**：根据距离赋予不同权重，提升预测精度。
- **特征选择/降维**：如 PCA、t-SNE，缓解维度灾难。
- **编辑近邻法（Edited Nearest Neighbor）**：去除噪声或冗余样本，压缩训练集。

