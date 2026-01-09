# src/engine/test_engine.py

import torch
import yaml
from torch import nn
from src.datasets import build_dataloader, build_cifar10_raw_dataloader
from src.models import build_model
from src.engine.train_engine import evaluate
from torchvision.utils import save_image
from src.utils.vis import to_pil

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_test(cfg_path):
    cfg = load_cfg(cfg_path)

    device = cfg.get("train", {}).get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = cfg["test"]["ckpt"]
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    _, _, test_loader = build_dataloader(cfg)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.2f}%")
    raw_loader = build_cifar10_raw_dataloader(
            data_root = str(cfg["dataset"]["data_root"]),
            batch_size = cfg["train"]["batch_size"],
            num_workers = 4,
            )
    images, labels = next(iter(raw_loader))
    img = images[0]  # PIL格式
    save_image(img, "outputs/tt.png")

    images1, labels1 = next(iter(test_loader))
    img1 = to_pil(images1[0])
    img1.save("outputs/tt1.png")
