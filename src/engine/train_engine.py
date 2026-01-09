# src/engine/train_engine.py

import os
import yaml

import torch
from torch import nn, optim

from src.models import build_model
from src.datasets import build_dataloader
from src.utils.seed import set_seed


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# 训练一个epoch  + 验证
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)  # (B, num_classes)
        total_corrects += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples else 0.0
    acc = 100.0 * total_corrects / total_samples if total_samples else 0.0

    return avg_loss, acc

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):  # 评估和训练区别是不需要优化器
    model.eval()
    total_loss = 0.0
    total_corrects = 0
    total_samples = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_corrects += (preds == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples else 0.0
    acc = 100.0 * total_corrects / total_samples if total_samples else 0.0

    return avg_loss, acc

# 保存最优模型
def save_checkpoint(path, model, optimizer, epoch, best_acc, class_names,):
    checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "class_names": class_names,
            }
    torch.save(checkpoint, path)


# 训练流程
def run_training(model, train_loader, val_loader, test_loader,
                 criterion, optimizer, device,
                 epochs, best_path, last_path, class_names):

    best_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_avg_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                )
        val_avg_loss, val_acc = evaluate(
                model, val_loader, criterion, device,
                )
        msg = (
                f"[Epoch {epoch:02d}/{epochs:02d}] |"
                f"train_loss={train_avg_loss:.2f}, train_acc={train_acc:.2f}% |"
                f"val_loss={val_avg_loss:.2f}, val_acc={val_acc:.2f}% |"
                )
        print(msg)

        save_checkpoint(last_path, model, optimizer, epoch, best_acc, class_names)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(best_path, model, optimizer, epoch, best_acc, class_names)

    test_avg_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[Test] loss={test_avg_loss:.2f}, acc={test_acc:.2f}%")

def run_train(cfg_path):
    cfg = load_cfg(cfg_path)

    # 随即种子
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Use device {device}")

    # 训练准备
    model = build_model(cfg).to(device)
    train_loader, val_loader, test_loader = build_dataloader(cfg)
    train_cfg = cfg["train"]
    lr = float(train_cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = int(train_cfg["epochs"])
    output_dir = str(train_cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    best_path = os.path.join(output_dir, "best.pt")
    last_path = os.path.join(output_dir, "last.pt")

    # 从数据集中取出类别数目和类别信息
    class_names = train_loader.dataset.dataset.classes

    # 开始训练
    run_training(
            model = model,
            train_loader = train_loader,
            val_loader = val_loader,
            test_loader = test_loader,
            criterion = criterion,
            optimizer = optimizer,
            device = device,
            epochs = epochs,
            best_path = best_path,
            last_path = last_path,
            class_names = class_names,
            )
