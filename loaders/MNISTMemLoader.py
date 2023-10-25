import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def scale_tensor(input_tensor, min_val, max_val):
    min_tensor = input_tensor.min()
    range_tensor = input_tensor.max() - min_tensor
    if range_tensor > 0:
        normalized_tensor = (input_tensor - min_tensor) / range_tensor
    else:
        normalized_tensor = torch.zeros(input_tensor.size())
    return min_val + (max_val - min_val) * normalized_tensor


class InMemoryMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, output_popcount, transform=None, binary=False):
        self.data = data
        self.targets = targets
        self.transform = transform
        self.binary = binary
        self.output_popcount = output_popcount

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        # if self.binary:
        #     x = scale_tensor(x, 0, 255)
        #     # Convert to uint8
        #     x = x.type(torch.uint8)
        #     # Convert each bit into a boolean tensor
        #     x = torch.from_numpy(np.unpackbits(x.numpy()).astype(np.bool))

        # if self.binary:
        #     min = torch.min(x)
        #     x = x > min
        #     x = x.view(28 * 28)

        return x, y

    def __len__(self):
        return len(self.data)


class FlattenTransform:
    def __call__(self, inputs):
        return inputs.view(-1, 28, 28)


def load_mnist_dataset(num_classes, output_popcount, binary=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        FlattenTransform(),
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_data = []
    train_targets = []

    for image, label in train_dataset:

        if binary:

            image_min = torch.min(image)
            train_image = torch.where(image > image_min, torch.ones_like(image, dtype=torch.float), torch.zeros_like(image, dtype=torch.float))
            train_image = train_image.view(28 * 28)

            train_data.append(train_image)

            # Create a tensor of size (num_classes*32,) filled with False
            # target_tensor = torch.full((num_classes * output_popcount,), False, dtype=torch.float)
            # Set the corresponding output_popcount bits of the correct class to True
            # target_tensor[label * output_popcount:(label + 1) * output_popcount] = 1.0
            # train_targets.append(target_tensor)

        else:
            train_data.append(image)

        train_targets.append(torch.eye(num_classes, dtype=torch.float)[label])

    train_data = torch.stack(train_data)
    train_targets = torch.stack(train_targets)

    test_data = []
    test_targets = []

    for image, label in test_dataset:

        if binary:
            image_min = torch.min(image)
            test_image = torch.where(image > image_min, torch.ones_like(image, dtype=torch.float), torch.zeros_like(image, dtype=torch.float))
            test_image = test_image.view(28 * 28)
            test_data.append(test_image)

            # Create a tensor of size (num_classes*output_popcount,) filled with False
            # target_tensor = torch.full((num_classes * output_popcount,), False, dtype=torch.float)
            # Set the corresponding 32 bits of the correct class to True
            # target_tensor[label * output_popcount:(label + 1) * output_popcount] = 1.0
            # test_targets.append(target_tensor)
        else:
            test_data.append(image)

        test_targets.append(torch.eye(num_classes, dtype=torch.float)[label])

    test_data = torch.stack(test_data)
    test_targets = torch.stack(test_targets)

    train_dataset = InMemoryMNISTDataset(train_data, train_targets, output_popcount, binary=binary)
    test_dataset = InMemoryMNISTDataset(test_data, test_targets, output_popcount, binary=binary)

    print('MNIST dataset loaded and transformed successfully!')
    return train_dataset, test_dataset


def get_loaders(num_classes, batch_size, validation_batch_size, output_popcount, binary=False):
    train_dataset, test_dataset = load_mnist_dataset(num_classes, output_popcount, binary=binary)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    train_results_loader = DataLoader(train_dataset, batch_size=validation_batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=validation_batch_size, shuffle=False, num_workers=8)
    return train_loader, test_loader, train_results_loader
