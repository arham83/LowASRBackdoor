from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.aggregate_block.dataset_and_transform_generate import get_dataset_normalization



def build_transforms(cfg):
    norm = get_dataset_normalization(cfg["dataset"])

    train_transform = transforms.Compose([
        transforms.Resize((cfg["input_height"], cfg["input_width"])),
        transforms.RandomCrop(
            cfg["input_height"],
            padding=cfg["random_crop_padding"]
        ),
        transforms.ToTensor(),
        norm,
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    return train_transform, test_transform


def build_dataloaders(cfg):
    train_transform, test_transform = build_transforms(cfg)

    if cfg["dataset"] != "mnist":
        raise ValueError("This script currently supports only MNIST.")

    train_dataset = datasets.MNIST(
        root=cfg["dataset_path"],
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=cfg["dataset_path"],
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    return train_dataset, test_dataset, train_loader, test_loader




def unnormalize_mnist(x, cfg):
    return x * cfg["std"] + cfg["mean"]