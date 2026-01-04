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
