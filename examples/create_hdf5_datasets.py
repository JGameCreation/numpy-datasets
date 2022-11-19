import numpy_datasets as nds
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pathlib


if __name__ == "__main__":

    workers = 64
    dataset_path = pathlib.Path("../../DATASETS/")

    tinet = ImageFolder(
        "/datasets01/tinyimagenet/081318/val", transform=transforms.ToTensor()
    )
    nds.loader.dataset_to_h5(
        tinet,
        dataset_path / "TINY_IMAGENET/val.hdf5",
        num_workers=workers,
        chunk_size=4096,
    )

    tinet = ImageFolder(
        "/datasets01/tinyimagenet/081318/train", transform=transforms.ToTensor()
    )
    nds.loader.dataset_to_h5(
        tinet,
        dataset_path / "TINY_IMAGENET/train.hdf5",
        num_workers=workers,
        chunk_size=4096,
    )

    # tinet = ImageFolder("/datasets01/imagenet_full_size/061417/val")
    # nds.loader.dataset_to_h5(
    #     tinet, dataset_path / "IMAGENET/val.hdf5", num_workers=workers, chunk_size=4096
    # )
