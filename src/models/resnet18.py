# src/models/resnet.py

import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    """
    针对CIFAR-10数据集全局微调ResNet18
    修改入口适配32 * 32图像,然后修改输出层
    """
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # 是否在预训练的权重之上进行微调
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # 修改入口处的7*7卷积和最大池化
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.model.maxpool = nn.Identity()

        # 修改出口的类别数目
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        
        return x
