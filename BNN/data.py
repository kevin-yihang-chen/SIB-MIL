import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pathlib
import tarfile
import requests
import shutil

from tqdm import tqdm

from models import MLP


def load_data(config_data, batch_size=32, image_size=32):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if config_data["name"] == "MNIST":
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif config_data["name"] == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif config_data["name"] == "CIFAR100":
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    elif config_data["name"] == "simulated":
        n = config_data["n"]
        scenario = config_data["scenario"]
        train_set, test_set = simulate(n, scenario)

    elif config_data["name"] == "TinyImageNet":
        train_set = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
        test_set = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)
        # Dataloaders
    elif config_data["name"] == "ImageNet":
        train_set = ImageNetV2Dataset(variant="matched-frequency", location="./data")
        test_set = ImageNetV2Dataset(variant="val", location="./data")
    else:
        raise NotImplementedError()

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_loader


def load_uncertainty_data(name, train, image_size, in_channel):
    if in_channel == 3:
        norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        norm = transforms.Normalize((0.1307,), (0.3081,))
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=in_channel),
            torchvision.transforms.ToTensor(),
            norm,
            torchvision.transforms.Resize(image_size)
        ])
    if name == "MNIST":
        d = torchvision.datasets.MNIST
    elif name == "FashionMNIST":
        d = torchvision.datasets.FashionMNIST
    elif name == "CIFAR10":
        d = torchvision.datasets.CIFAR10
    elif name == "CIFAR100":
        d = torchvision.datasets.CIFAR100
    elif name == "TinyImageNet":
        if train:
            return torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=transform)
        else:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Grayscale(num_output_channels=in_channel),
                    torchvision.transforms.ToTensor(),
                    norm,
                    torchvision.transforms.Resize((image_size, image_size))
                ])
            return torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=transform)
    elif name == "OMIGLOT":
        d = torchvision.datasets.Omniglot
        return d(root='./data', background=train, download=True, transform=transform)
    elif name == "SVHN":
        train = "train" if train else "test"
        d = torchvision.datasets.SVHN
        return d(root='./data', split=train, download=True, transform=transform)
    else:
        raise NotImplementedError

    return d(root='./data', train=train, download=True, transform=transform)


def simulate(n, scenario):
    dataset = SimulatedDataset(n, scenario)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    return trainset, testset


class SimulatedDataset(Dataset):
    def __init__(self, n, scenario, mean=0, sd=1):
        # Set random seed
        np.random.seed(30)
        self.n = n
        if scenario == 1:  # Polynomial case
            x = np.random.uniform(-5, 5, size=(1, n))
            eps = np.random.normal(loc=mean, scale=3, size=n)
            self.y = x[0] ** 3 + eps
        elif scenario == 2:  # Linear Regression
            x = np.random.uniform(-5, 5, size=(4, n))
            self.y = x[0] + 1 /4 * x[1] + 2 * x[2] ** 2 + x[3]
        elif scenario == 3:  # Non linear case
            x = np.random.uniform(-5, 5, size=(4, n))
            eps = np.random.normal(loc=mean, scale=3, size=n)
            self.y = x[0] * (x[1] ** 2 + 1) + x[2] * x[3] + eps
        elif scenario == 4:  # Sparse case
            p = 1000
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p * 9 // 10)
            beta_2 = np.random.uniform(0, 1, size=p // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
        elif scenario == 5:  # Dense case
            p = 1000
            x = np.random.uniform(-5, 5, size=(p, n))
            eps = np.random.normal(loc=mean, scale=2, size=n)
            beta_1 = np.zeros(p // 10)
            beta_2 = np.random.uniform(0, 1, size=p * 9 // 10)
            beta = np.concatenate([beta_1, beta_2])
            beta = np.random.permutation(beta)
            self.y = (np.einsum("ij, i -> j", x, beta) + eps)
        elif scenario == 6:  # Neural network case
            p = 1000
            net = MLP(1, p, "Freq", n_blocks=2)
            x = np.random.uniform(-5, 5, size=(p, n)).astype(np.float32)
            eps = np.random.normal(loc=mean, scale=2, size=n)

            self.y = net(torch.from_numpy(x.T))
            self.y = self.y.detach().numpy().squeeze() + eps
        else:
            raise NotImplementedError("This scenario is not implemented")

        self.y = self.y.astype('float32')

        assert x is not None

        self.covariates = x.T.astype('float32')

    def __len__(self):
        return self.covariates.shape[0]

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.covariates[idx], self.y[idx]

URLS = {"matched-frequency" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
        "val": "https://imagenet2val.s3.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}


V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000


class ImageNetValDataset(Dataset):
    def __init__(self, transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/imagenet_validation/")
        self.tar_root = pathlib.Path(f"{location}/imagenet_validation.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.JPEG"))
        self.transform = transform
        if not self.dataset_root.exists() or len(self.fnames) != VAL_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset imagenet-val not found on disk, downloading....")
                response = requests.get(URLS["val"], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES['val']}", self.dataset_root)

        self.dataset = ImageFolder(self.dataset_root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img, label = self.dataset[i]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/ImageNetV2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/ImageNetV2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            if not self.tar_root.exists():
                print(f"Dataset {variant} not found on disk, downloading....")
                response = requests.get(URLS[variant], stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                with open(self.tar_root, 'wb') as f:
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        f.write(data)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    assert False, f"Downloading from {URLS[variant]} failed"
            print("Extracting....")
            tarfile.open(self.tar_root).extractall(f"{location}")
            shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
