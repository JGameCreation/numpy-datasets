import numpy_datasets as nds
import torch as ch
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm


if __name__ == "__main__":
    # mnist = CIFAR10("../../DATASETS/CIFAR10", download=False, train=True)
    # nds.loader.dataset_to_h5(mnist, "../../DATASETS/CIFAR10/train.hdf5", num_workers=32)

    # tinet = ImageFolder("/datasets01/tinyimagenet/081318/val")
    # nds.loader.dataset_to_h5(
    #     tinet, "../../DATASETS/TINY_IMAGENET/val.hdf5", num_workers=16
    # )

    C = 256
    BS = 256
    dataset = nds.loader.H5Dataset(
        "../../DATASETS/TINY_IMAGENET/val.hdf5", device="cuda", chunkit=C
    )

    print(len(dataset))

    loader = iter(
        DataLoader(
            dataset,
            batch_size=BS // C,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
    )

    t = time()
    # for (x,y) in loader:
    #     x.to_
    current = nds.loader.mapit(next(loader))
    ahead = nds.loader.mapit(next(loader))

    for i in tqdm(range(len(loader) - 2)):
        current[0].max()
        current, ahead = ahead, nds.loader.mapit(next(loader))

    # mnist = CIFAR10(
    #     "../../DATASETS/CIFAR10", download=False, train=True, transform=ToTensor()
    # )
    mnist = ImageFolder(
        "/datasets01/tinyimagenet/081318/val",
        transform=ToTensor(),
    )
    loader = DataLoader(
        mnist, batch_size=BS, shuffle=True, num_workers=1, pin_memory=True
    )
    t = time()
    for (x, y) in tqdm(loader):
        x = x.to(device="cuda", non_blocking=True)
        x.max()
