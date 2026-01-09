# src/datasets/cifar10.py
"""
从本地加载train,eval,test数据
"""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision import transforms as T


class CIFAR10Transforms:
    IMAGE_SIZE = 32
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)

    @classmethod
    def train(cls):
        return T.Compose([
            T.Resize((cls.IMAGE_SIZE, cls.IMAGE_SIZE)),
            T.RandomCrop(cls.IMAGE_SIZE, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(cls.MEAN, cls.STD),
            ])

    @classmethod
    def test(cls):
        return T.Compose([
            T.Resize((cls.IMAGE_SIZE, cls.IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(cls.MEAN, cls.STD),
            ])

def build_cifar10_datasets(data_root, val_ratio):

    # 官方的train数据集(分割成trian and eval)
    full_train = datasets.CIFAR10(
            root = data_root,
            train = True,
            download = False,
            transform = CIFAR10Transforms.train(),
            )

    # 按照比例划分
    total_size = len(full_train)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_set, val_set = random_split(
            full_train,
            [train_size, val_size],
            )

    # 官方的test集合
    test_set = datasets.CIFAR10(
            root = data_root,
            train = False,
            download = False,
            transform = CIFAR10Transforms.test(),
            )

    return train_set, val_set, test_set

def build_cifar10_dataloaders(data_root, batch_size, val_ratio, num_workers, pin_memory, shuffle):

    train_set, val_set, test_set = build_cifar10_datasets(
            data_root = data_root,
            val_ratio = val_ratio,
            )

    train_loader = DataLoader(
            train_set,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            pin_memory = pin_memory,
            )

    val_loader = DataLoader(
            val_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
            )

    test_loader = DataLoader(
            test_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            pin_memory = pin_memory,
            )

    return train_loader, val_loader, test_loader

def build_cifar10_vis_dataloader(data_root, batch_size, num_workers):
    vis_set = datasets.CIFAR10(
            root = data_root,
            train = False,
            download = False,
            transform = T.ToTensor(),
            )

    vis_dataloader = DataLoader(
            vis_set,
            batch_size = batch_size,
            shuffle = False,
            num_workers = num_workers,
            )

    return vis_dataloader

def build_cifar10_raw_dataloader(data_root, batch_size, num_workers, train=False, shuffle=False):
    raw_set = datasets.CIFAR10(
            root = data_root,
            train=train,
            download = False,
            transform = T.ToTensor(),
            )

    raw_loader = DataLoader(
            raw_set,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = num_workers,
            )
    
    return raw_loader
