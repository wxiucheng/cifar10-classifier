# src/models/__init__.py
from .cnn import SimpleCNN
from .resnet18 import ResNet18

def build_model(cfg):
    """
    只通过cfg来控制模型
    """
    model_cfg = cfg["model"]
    name = model_cfg["name"]
    num_classes = int(model_cfg["num_classes"])

    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)

    if name == "resnet18":
        pretrained = bool(model_cfg["pretrained"])
        return ResNet18(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported model {name}")

__all__ = ["build_model", "SimpleCNN", "ResNet18"]

