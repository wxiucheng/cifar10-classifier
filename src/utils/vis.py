# src/utils/vis.py
import torch
from PIL import Image

# CIFAR10 固定
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)


def denorm(x: torch.Tensor) -> torch.Tensor:
    """
    x: (3,H,W) or (B,3,H,W)  (Normalize 后)
    return: 同形状，反归一化到 [0,1]
    """
    mean = torch.tensor(MEAN, device=x.device, dtype=x.dtype)
    std  = torch.tensor(STD,  device=x.device, dtype=x.dtype)

    if x.dim() == 3:
        mean = mean.view(3, 1, 1)
        std  = std.view(3, 1, 1)
    elif x.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std  = std.view(1, 3, 1, 1)
    else:
        raise ValueError(f"Expected x dim 3 or 4, got {x.dim()}")

    return (x * std + mean).clamp(0.0, 1.0)


def to_pil(x: torch.Tensor) -> Image.Image:
    """
    单张：(3,H,W) Normalize 后 -> PIL
    """
    if x.dim() != 3:
        raise ValueError(f"to_pil expects (3,H,W), got {tuple(x.shape)}")

    y = denorm(x).detach().cpu()
    y = (y * 255.0).round().to(torch.uint8)  # (3,H,W)
    y = y.permute(1, 2, 0).numpy()           # (H,W,3)
    return Image.fromarray(y, mode="RGB")
