# src/datasets/__init__.py

from .cifar10 import build_cifar10_dataloaders, CIFAR10Transforms, build_cifar10_vis_dataloader,  build_cifar10_raw_dataloader
from .cifar100 import build_cifar100_dataloaders, CIFAR100Transforms


def build_dataloader(cfg):
    ds_cfg = cfg["dataset"]
    train_cfg = cfg["train"]

    name = ds_cfg["name"]

    if name == "cifar10":
        return build_cifar10_dataloaders(
                data_root = ds_cfg["data_root"],
                val_ratio = ds_cfg["val_ratio"],
                batch_size = train_cfg["batch_size"],
                num_workers = ds_cfg.get("num_workers", 4),
                pin_memory = ds_cfg.get("pin_memory", True),
                shuffle = ds_cfg.get("shuffle", True),
                )
    elif name == "cifar100":
        return build_cifar100_dataloaders(
                data_root = ds_cfg["data_root"],
                val_ratio = ds_cfg.get("val_ratio", 0.1),
                batch_size = train_cfg["batch_size"],
                num_workers = ds_cfg.get("num_workers", 4),
                pin_memory = ds_cfg.get("pin_memory", True),
                shuffle = ds_cfg.get("shuffle", True),
                )
    else:
        raise ValueError(f"Unknown dataset name:{name}")

__all__ = [
        "build_dataloader",
        "build_cifar10_vis_dataloader",
        "CIFAR10Transforms",
        "CIFAR100Transforms",
        "build_cifar10_raw_dataloader"
        ]
