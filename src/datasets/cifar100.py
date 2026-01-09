# src/datasets/cifar100.py

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms as T


class CIFAR100Transforms:
    IMAGE_SIZE = 32
    MEAN = (0.5071, 0.4867, 0.4408)
    STD = (0.2675, 0.2565, 0.2761)

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

def build_cifar100_datasets(data_root, val_ratio):
    
    # 官方的train
    full_train = datasets.CIFAR100(
            root = data_root,
            train = True,
            download = False,
            transform = CIFAR100Transforms.train(),
            )
    
    # 按比例分割
    total_size = len(full_train)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    train_set, val_set = random_split(
            full_train,
            [train_size, val_size],
            )

    # 官方的test
    test_set = datasets.CIFAR100(
            root = data_root,
            train = False,
            download = False,
            transform = CIFAR100Transforms.test(),
            )

    return train_set, val_set, test_set

def build_cifar100_dataloaders(data_root, val_ratio,batch_size, num_workers, pin_memory, shuffle):

    train_set, val_set, test_set = build_cifar100_datasets(
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
