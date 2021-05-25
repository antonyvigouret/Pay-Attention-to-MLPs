import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.transforms.transforms import Resize

CIFAR_DEFAULT_MEAN = (0.491, 0.482, 0.446)
CIFAR_DEFAULT_STD = (0.247, 0.243, 0.261)


def get_loaders(batch_size=4, eval_split=0.15):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(CIFAR_DEFAULT_MEAN, CIFAR_DEFAULT_STD),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    len_trainset = len(train_dataset)
    train_dataset, eval_dataset = random_split(
        train_dataset, [int(len_trainset * (1 - eval_split)), int(len_trainset * eval_split)]
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=2)

    return train_loader, eval_loader, test_loader
