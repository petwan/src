---
title: ⚡NVIDIA PilotNet 学习笔记
date: 2026-01-07
tags: [Pytorch]
description: 主要介绍NVIDIA PilotNet自动驾驶汽车端到端学习方法，以及使用Pytorch和公开数据进行类似效果的训练。
draft: false
---

# ⚡NVIDIA PilotNet 学习笔记
## 1. 论文学习
### 1.1 背景

2016年 NVIDIA发表了论文《**End to End Learning for Self-Driving Cars**》，训练了一个CNNs，将车前部摄像头捕捉到的原始像素图映射为汽车的方向操控命令。

与驾驶问题的显式分解方法（如车道标志检测、路径规划和控制）相比，端到端系统可以同时对所有处理步骤（processing steps）进行优化。

:::info
相比之下，传统方法中针对人工选择的中间准则（criteria）进行优化的方法，如车道检测，容易理解选择这样的criteria非常容易进行人工解释，但这并不能自动的保证实现最大的系统性能。
:::

数据采集系统记录挡风玻璃后面三台摄像机的数据 和 驾驶员操控方向盘的偏转角度，摄像机生成的每一帧视频数据（30FPS）都与人类驾驶员的转向角度进行时间同步。

<Image 
src="assets/nvidia_pilotnet_data_collection_system.png"
width="80%"
card="true"
 />

为了使系统能够独立于汽车的几何尺寸，采用 1/r 来表示方向控制命令，其中 r 是以米为单位的转弯半径。

:::info
使用 1/r 而不是 r 的目的是防止在直线行驶时出现奇点（直线行驶的转弯半径无限大）。左转弯的 1/r 值为负数，右转弯的值为正数。
:::

但是只有来自人类驾驶员的正确数据是不足以完成训练的，神经网络还必须学习如何从任何错误中恢复，否则自动驾驶汽车就将慢慢偏移道路。因此训练数据还扩充了额外的图像，这些图像显示了远离车道中心的偏离程度以及不同道路方向上的转动。两个特定偏离中心的变化图像可由左右两个摄像机捕获。训练数据准备完毕之后，将其送入一个卷积神经网络（CNN）

<Image 
src="assets/training_the_neural_network.png"
width="80%"
card="true"
/>

预测的方向控制命令与理想的控制命令相比较，然后调整CNN模型的权值使得预测值尽可能接近理想值。

在这个框架中，只要提供足够的训练数据，即人类驾驶员驾驶携带有摄像头的车辆累计驾驶大量的里程，再加上人为创造系统的“极限”道路状态——偏离道路线的各种工况，CNN就会得到充分的训练，而变得足够强大。一旦训练完成，网络就能够从单中心摄像机（single center camera）的视频图像中生成转向命令

<Image
src="assets/trained_network.png" 
width="80%"
card="true"
/>

### 1.2 模型
训练网络的权重值，使得网络模型输出的方向控制命令与人工驾驶或者调整后的控制命令的均方误差最小。使用到的网络一共包括9层，包括一个 normalization 层，五个 convolutional 层和三个 fully connected 层。输入图像被映射到YUV平面，然后传入网络。

<Image
src="assets/nvidia_pilotnet_network_architecture.png"
card="true"
/>
最终的输出是是转弯半径的倒数

### 1.3 数据选择
训练神经网络的第一步就是选择使用视频的哪些帧。采集的数据标记了道路类型、天气条件、驾驶员行为（保持车道行驶、更换车道、转弯等等）等标签。

用CNN训练保持车道的行驶，这里只挑选驾驶员保持同一车道行驶的数据，抛弃剩余部分。同时以10FPS对视频降采样，因为用高采样率得到的图像相似度非常高，并不能带来很多有用的信息。为了消除直线行驶的偏置，很大一部分训练图像包含了有弧度的道路。

选定最终的图像集之后，我们人工添加了一些偏移和旋转来补充数据，教会网络如何修复较差的姿势和视角。调整的幅度按照正态分布随机选取。分布的均值为零，标准差是驾驶员操作数据的标准差的两倍。

### 1.4 仿真
在上路测试训练得到的CNN之前，我们首先仿真测试网络的性能。

simulator发送所选测试视频的第一帧，调整真值的偏离（因为采集过程中并不是一直沿着中心线开的，可能稍微偏左或偏右，这里要将真值进行调整），然后将输入给到训练的CNN模型，CNN之后输出这帧的转向信号。CNN转向信号以及采集到的驾驶员转向一起喂到车辆动态模型中，以更新仿真车辆的位置和朝向

仿真器会根据车辆新的位置和朝向，更新测试视频中下一帧的图片,仿真器会记录车辆到车道中心线的距离，yaw以及虚拟车行驶过的距离，当车辆到车道中心线的距离超过一定的阈值，虚拟驾驶员进行干预，虚拟车的位置和朝向会重设到原测试视频的真值位置

仿真器采用预先用数据采集车的前置摄像机录制的视频数据，然后根据图像用CNN模型预测操控命令。这些录制视频的时间轴与驾驶员操控命令的时间轴保持一致。由于驾驶员不总是将车辆保持在车道的中心，我们必须人工校准车道的中心，因为它与模拟器所采用的视频的每一帧都关联。我们把这个姿态称为“对照数据”。

通过计算虚拟人干预车辆行驶的次数来估计网络模型能够自动驾驶汽车的时间段占比。我们假设在现实情况下发生一次人工干预需要六秒：这个时间包括驾驶员接管操纵车辆、纠正车辆位置、重新进入自动驾驶模式。

$$
autonomy = (1-\frac{number\_of\_interventions * 6 seconds}{elapsed\_time [seconds]}) * 100
$$

如果自600秒内触发了10次，则自动驾驶的比例计算为

$$
autonomy = (1-\frac{10*6}{600})* 100 = 90\%
$$

### 1.5 道路测试
实际道路测试时间不包括变换车道和转弯，仅仅看车道居中，https://youtu.be/NJU9ULQUwng

## 2. 总结
作者经验表明，神经网络能学习完整的保持车道驾驶的任务，而不需要人工将任务分解为道路和车道检测、语义抽象、道路规划和控制。

从不到一百小时的少量训练数据就足以训练在各种条件下操控车辆，比如在高速公路、普通公路和居民区道路，以及晴天、多云和雨天等天气状况。

CNN模型可以从非常稀疏的训练信号（只有方向控制命令）中学到有意义的道路特征。例如，系统不需要标注数据就学会了识别道路边界。

## 3. 个人笔记<Badge type="tip" text="仅供参考" />
### 3.1 PilotNet
PilotNet 是一个单一的深度神经网络 (DNN)，它以像素为输入，并生成期望的方向盘转角作为输出。

模型输入是一个 200 x 66 的单张图像，输出方向盘转角。其结构是5层卷积 + 4层全连接，激活函数为ReLU（最后的输出层无激活）。

在PilotNet中，左右两侧的摄像头用于提供增强的训练数据。这些摄像头记录的视角与中央摄像头的视角相同，相当于车辆在车道上向左或向右偏移了一定距离。此外，所有摄像头图像都经过一定范围的视角变换，从而生成对应于任意偏移的数据。这些偏移后的图像与转向指令配对，引导车辆回到车道中央。

早期的PilotNet系统输出的是转向角。虽然这种方法简单易行，但也存在一些缺陷。首先，转向角并不能唯一确定车辆的实际行驶路径。实际路径取决于多种因素，例如车辆动力学、车辆几何形状、路面状况以及道路的横向坡度（例如路堤或隆起）。忽略这些因素会导致车辆偏离车道中心。此外，仅输出当前转向角无法提供车辆可能行驶的路线（意图）信息。最后，仅输出转向角的系统难以与障碍物检测系统集成。

::: info
为了克服这些局限性，PilotNet 现在输出相对于车辆坐标系的三维坐标系中的期望轨迹。然后，一个独立的控制器引导车辆沿该轨迹行驶。与仅仅跟随转向角不同，跟随轨迹行驶使得 PilotNet 能够保持一致的行为，而不受车辆特性的影响。

使用轨迹作为输出的一个好处是，它有助于将 PilotNet 的输出与来自其他来源的信息（例如地图或单独的障碍物检测系统）融合。

为了训练这个新版 PilotNet，需要创建期望的目标轨迹。使用人工标注员确定的车道中心线作为真实期望轨迹。人工标注员还会在没有划线的道路上指定中心线，使 PilotNet 能够在这些情况下预测车道边界。车道边界的输出允许第三方使用 PilotNet 提供的路径感知来规划他们自己的轨迹。

早期的PilotNet采用了典型的卷积神经网络结构。而较新的版本则采用了改进的ResNet结构。**这里仅使用早期经典的卷积神经网络结构**
:::

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PilotNet(nn.Module):
    """
    NVIDIA PilotNet for end-to-end steering angle prediction.

    Input:  (B, 3, H, W)   e.g., (B, 3, 66, 200) or (B, 3, 160, 320)
    Output: (B, 1)          predicted steering angle
    """

    def __init__(self, input_shape=(3, 66, 200), output_dim=1):
        super(PilotNet, self).__init__()
        C, H, W = input_shape

        # Convolutional layers
        self.conv1 = nn.Conv2d(C, 24, kernel_size=5, stride=2, padding=0)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=0)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=0)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Compute size after convolutions (for linear layer input)
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = self._forward_conv(dummy)
            self.fc_input_dim = x.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # no activation on final layer
        return x


# === 示例用法 ===
if __name__ == "__main__":
    # 创建模型（使用原始论文推荐的输入尺寸：66x200）
    model = PilotNet(input_shape=(3, 66, 200))

    # 模拟输入：batch_size=4 的图像
    dummy_input = torch.randn(4, 3, 66, 200)

    # 前向传播
    output = model(dummy_input)
    print("Output shape:", output.shape)  # torch.Size([4, 1])
```

### 3.2 Dataset
Nvidia 并未提供其采集到的数据集，因此这里采用了开源的[driving_dataset](https://github.com/SullyChen/driving-datasets?tab=readme-ov-file)

采集到的图像大小为 455×256，因此需要进行裁剪和resize，主要的裁切是把图像的底部一定范围截掉，把顶部天空部分裁掉，然后resize为 200×66。

```python
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random


def _load_data(data_dir="driving_dataset", test_size=0.2, random_state=42):
    xs, ys = [], []
    data_file = os.path.join(data_dir, "data.txt")

    with open(data_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            img_path = os.path.join(data_dir, "data", parts[0])
            angle_deg = float(parts[1].split(",")[0])
            angle_rad = angle_deg * np.pi / 180.0
            xs.append(img_path)
            ys.append(angle_rad)

    combined = list(zip(xs, ys))
    random.seed(random_state)
    random.shuffle(combined)
    xs[:], ys[:] = zip(*combined)

    split_idx = int(len(xs) * (1 - test_size))
    train_xs, val_xs = xs[:split_idx], xs[split_idx:]
    train_ys, val_ys = ys[:split_idx], ys[split_idx:]

    return (train_xs, train_ys), (val_xs, val_ys)


(train_paths, train_angles), (val_paths, val_angles) = _load_data()


class DrivingDataset(Dataset):
    def __init__(self, train=True):
        self.paths = train_paths if train else val_paths
        self.angles = train_angles if train else val_angles

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        angle = self.angles[idx]

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # 原图尺寸: 455x256 (WxH)
        h, w = image.shape[:2]  # h=256, w=455

        # 按比例裁剪道路区域（参考 NVIDIA 的裁剪比例）
        x_start = int(0.0 * w)  # 0
        x_end = int(1.0 * w)  # 455
        y_start = int(0.15 * h)
        y_end = int(0.75 * h)

        cropped = image[y_start:y_end, x_start:x_end]

        # Resize to (200, 66)
        resized = cv2.resize(cropped, (200, 66))  # (66, 200, 3)

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1))

        return torch.from_numpy(chw), torch.tensor(angle, dtype=torch.float32)
```

### 3.3 Training
之前是基于 keras实现，更新为基于 Pytorch 的实现。
```python
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import PilotNet
from driving_dataset import DrivingDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PilotNet().to(device)

    batch_size = 32
    num_workers = 4

    train_dataset = DrivingDataset(train=True)
    val_dataset = DrivingDataset(train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(data).squeeze(1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = model(data).squeeze(1)
                val_loss += criterion(output, target).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                "best_pilotnet.pth",
            )
            print(f"→ Saved best model with val_loss = {best_val_loss:.6f}")

    print("Training finished.")


if __name__ == "__main__":
    train()
```

### 3.4 理解模型
使用输入图像的绝对值最大通道来生成注意力热力图，突出显示模型做转向决策时关注的区域。

- 加载训练好的 best_pilotnet.pth
- 读取一张原始 455×256 图像
- 应用与训练时 完全一致的预处理（裁剪 + resize）
- 计算梯度显著性图
- 将显著性图 反投影回原始图像尺寸（便于理解）
- 可视化：原图 + 显著性叠加图

<Image
src="assets/saliency_result.png"
/>

再看一个右转的例子，10010.jpg
<Image
src="assets/saliency_result_10010.png"
/>

## References
1. [https://arxiv.org/pdf/1604.07316.pdf](https://arxiv.org/pdf/1604.07316.pdf)

2. [Explaining How a Deep Neural Network Trained with End-to-End Learning Steers a Car](https://arxiv.org/abs/1704.07911)

3. [driving_dataset](https://github.com/SullyChen/driving-datasets?tab=readme-ov-file)
