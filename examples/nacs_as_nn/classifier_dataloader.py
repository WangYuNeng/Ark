import torch
import torch.utils
import torch.utils.data
import torchvision
from classifier_parser import parser
from torch.utils.data import DataLoader

args = parser.parse_args()
img_noise_std = args.img_noise_std


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.2):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        if self.std == 0:
            return tensor
        return torch.clip(
            tensor + torch.randn(tensor.size()) * self.std + self.mean,
            min=-1.0,
            max=1.0,
        )

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


normalize = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # Normalize to [-1, 1]
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        # Squeez the channel dimension
        torchvision.transforms.Lambda(lambda x: x.squeeze(0)),
        # Convert to float64
        torchvision.transforms.Lambda(lambda x: x.double()),
        # Add Gaussian noise
        AddGaussianNoise(0.0, img_noise_std),
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
    return dataloader, val_dataloader
