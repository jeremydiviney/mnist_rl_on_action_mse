import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class InMemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class OneHotEncodedCIFAR10(datasets.CIFAR10):
    def __init__(self, *args, num_classes=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert num_classes is not None, "Number of classes must be provided for one-hot encoding"
        self.num_classes = num_classes

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        one_hot_target = torch.zeros(self.num_classes, dtype=torch.float)
        one_hot_target[target] = 1.0
        return img, one_hot_target


def load_cifar10_dataset(batch_size, num_classes, num_workers=8):
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),  # add random cropping
        # transforms.RandomHorizontalFlip(),  # add random horizontal flipping
        # transforms.RandomRotation(10),  # add random rotation
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # add auto augmentation
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # normalize data
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # normalize data
    ])

    train_dataset = OneHotEncodedCIFAR10('./data', train=True, download=True, transform=transform_train, num_classes=num_classes)
    test_dataset = OneHotEncodedCIFAR10('./data', train=False, download=True, transform=transform_test, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, pin_memory=True)

    print('CIFAR-10 dataset loaded and transformed successfully!')
    return train_loader, test_loader


def get_loaders(num_classes, batch_size):
    train_loader, test_loader = load_cifar10_dataset(batch_size, num_classes)
    return train_loader, test_loader
