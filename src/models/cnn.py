# src/models/cnn.py
import torch.nn as nn


class SimpleCNN(nn.Module):
    """
    一个很简单的CNN,用于CIFAR-10(3, 32, 32)图片的分类
    """
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # (16, 16)

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # (8, 8)

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # (4, 4)
                nn.Flatten(),  # 默认从第1维度展平到最后,正好把Batch保留
                )

        self.classifier = nn.Sequential(
                nn.Linear(128 * 4 * 4, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
                )

    def forward(self, x):
        x = self.features(x)  # (B, 128 * 4 * 4)
        x = self.classifier(x)  # (B, num_classes)

        return x
