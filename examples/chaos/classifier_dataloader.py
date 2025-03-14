import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader

normalize = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # Normalize to [-1, 1]
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        # Flatten
        torchvision.transforms.Lambda(lambda x: x.view(-1)),
        # Convert to float64
        torchvision.transforms.Lambda(lambda x: x.double()),
    ]
)


def get_dataloader(
    dataset: str,
    batch_size: int,
    shuffle: bool,
    train: bool,
    validation_split: float,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "mnist":
        dataset = torchvision.datasets.MNIST(
            root="data",
            train=train,
            download=True,
            transform=normalize,
        )
    elif dataset == "fashion_mnist":
        dataset = torchvision.datasets.FashionMNIST(
            root="data",
            train=train,
            download=True,
            transform=normalize,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if validation_split > 0:
        n_train = int(len(dataset) * (1 - validation_split))
        n_val = len(dataset) - n_train
        dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
    else:
        val_dataset = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    if validation_split > 0:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        val_dataloader = None

    return dataloader, val_dataloader
