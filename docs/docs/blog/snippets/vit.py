import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 32/4=8 → 8*8=64 patches

        # 用 Conv2d 实现：滑动窗口大小=patch_size，步长=patch_size → 自动分块
        self.proj = nn.Conv2d(
            in_channels=in_channels,  # 输入通道数（RGB=3）
            out_channels=embed_dim,  # 输出维度（比如128）
            kernel_size=patch_size,  # 卷积核大小 = patch 大小
            stride=patch_size,  # 步长 = patch 大小 → 不重叠
        )

    def forward(self, x):
        # x 形状: [Batch, Channels, Height, Width] → 比如 [1, 3, 32, 32]
        x = self.proj(x)  # → [1, 128, 8, 8]
        x = x.flatten(2)  # 把最后两个维度展平 → [1, 128, 64]
        x = x.transpose(1, 2)  # 转置 → [1, 64, 128] （序列格式）
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

    def forward_with_attention(self, x):
        q = k = v = self.norm1(x)
        attn_out, attn_weights = self.attn(
            q, k, v, need_weights=True, average_attn_weights=False
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=6,  # Transformer 层数 # [!code ++]
        num_heads=4,  # 注意力头数 [!code ++]
        dropout=0.1,  # dropout [!code ++]
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token: 可学习的参数，形状 [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码：可学习的参数，形状 [1, num_patches+1, embed_dim]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # --- 新增：Transformer Encoder ---
        self.dropout = nn.Dropout(dropout)  # [!code ++]
        self.transformer_blocks = nn.ModuleList(
            [  # [!code ++]
                TransformerBlock(embed_dim, num_heads, dropout)
                for _ in range(depth)  # [!code ++]
            ]
        )  # [!code ++]

        # --- 新增：Classification Head --- # [!code ++]
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # --- 新增：通过 Transformer Encoder ---
        x = self.dropout(x)  # [!code ++]
        for block in self.transformer_blocks:  # [!code ++]
            x = block(x)  # [!code ++]
        return self.head(x[:, 0])  # 分类结果 # [!code ++]

    def forward_get_last_attention(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        for i, block in enumerate(self.transformer_blocks):
            if i < len(self.transformer_blocks) - 1:
                x = block(x)
            else:
                x, attn_weights = block.forward_with_attention(x)
        return attn_weights  # [B, N, N]


def get_dataloaders(batch_size=64):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader, testloader


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Val"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 超参数
    batch_size = 64
    epochs = 15
    lr = 3e-4

    # 数据
    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    # 模型
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    print(
        f"📊 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # 损失与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    best_acc = 0.0
    # 训练
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"📈 Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"📉 Val   Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "vit_cifar10.pth")
            print("✅ Best model saved as 'vit_cifar10.pth'")

    print("\n✅ Training completed!")


def visualize_attention(
    checkpoint_path="vit_cifar10.pth", idx=0, device=None, save_path="attention_vis.png"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        num_classes=10,
        embed_dim=128,
        depth=6,
        num_heads=4,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")

    # 获取单个测试样本
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    image, label = testset[idx]

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # 前向传播获取最后一层 attention
    with torch.no_grad():
        attn_weights = model.forward_get_last_attention(image.unsqueeze(0).to(device))

    # 提取 [CLS] token 对所有 patch 的 attention（平均所有头）
    cls_attn = attn_weights[0, :, 0, 1:].mean(dim=0).cpu()
    h_patches = w_patches = model.img_size // model.patch_size  # 32//4 = 8
    cls_attn_map = cls_attn.view(1, 1, h_patches, w_patches)  # [1, 1, 8, 8]

    # 使用 F.interpolate 上采样到原始图像大小 (32x32)
    cls_attn_up = (
        F.interpolate(
            cls_attn_map,
            size=(model.img_size, model.img_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .numpy()
    )  # [32, 32]

    # 反归一化图像用于显示
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    img_np = image.permute(1, 2, 0).numpy()
    img_unnorm = np.clip(img_np * std + mean, 0, 1)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_unnorm)
    axes[0].set_title(f"True Label: {class_names[label]}")
    axes[0].axis("off")

    im1 = axes[1].imshow(cls_attn.view(h_patches, w_patches), cmap="viridis")
    axes[1].set_title("[CLS] → Patches\n(Attention Grid)")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(img_unnorm)
    axes[2].imshow(cls_attn_up, cmap="jet", alpha=0.5)
    axes[2].set_title("Attention Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Attention visualization saved to '{save_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "vis"])
    parser.add_argument("--idx", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "train":
        main()
    elif args.mode == "vis":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        visualize_attention(
            checkpoint_path="vit_cifar10.pth", idx=args.idx, device=device
        )
