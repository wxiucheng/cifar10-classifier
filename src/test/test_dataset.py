import argparse
import os
import torch
import yaml
from torch import nn
from src.datasets import build_dataloader
from src.models import build_model
from src.train.train import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default= None,
        help="YAML配置文件路径",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="模型权重路径(默认使用output_dir/best.pt)",
    )
    return parser.parse_args()


def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_ckpt_path(cfg, ckpt_arg):
    if ckpt_arg:
        return ckpt_arg
    output_dir = cfg.get("train", {}).get("output_dir", "outputs")
    return os.path.join(output_dir, "best.pt")

def build_criterion(cfg):
    train_cfg = cfg.get("train", {})
    criterion_name = str(train_cfg.get("criterion", "cross_entropy")).lower()
    if criterion_name == "cross_entropy":
        label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported criterion {criterion_name}")


def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    device = cfg.get("train", {}).get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg)

    ckpt_path = resolve_ckpt_path(cfg, args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, _, test_loader = build_dataloader(cfg)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.2f}%")


if __name__ == "__main__":
    main()
