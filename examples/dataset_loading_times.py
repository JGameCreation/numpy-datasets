import numpy_datasets as nds
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch as ch
import torch.multiprocessing as mp


def launcher(local_rank, option, node_rank=0, gpus=2):
    world_size = gpus * 1
    global_rank = node_rank * gpus + local_rank
    C = 256
    BS = 256

    distributed = True
    if option == 0:
        dataset = nds.loader.H5Dataset(
            "../../DATASETS/TINY_IMAGENET/val.hdf5",
            device=f"cuda:{local_rank}",
            chunkit=C,
        )
    else:
        dataset = ImageFolder(
            "/datasets01/tinyimagenet/081318/val",
            transform=ToTensor(),
        )

    if distributed:
        ch.distributed.init_process_group(
            "nccl",
            rank=global_rank,
            world_size=world_size,
            init_method="tcp://127.0.0.1:23456",
        )
        ch.cuda.set_device(local_rank)

    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=global_rank)
        if distributed
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=BS // C if option == 0 else BS,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
    )

    for e in range(3):
        if distributed:
            sampler.set_epoch(e)
        if option == 0:
            for (x, y) in tqdm(nds.loader.FastLoader(loader)):
                x.max()
        else:
            for (x, y) in tqdm(loader):
                x = x.to(device="cuda", non_blocking=True)
                x.max()


if __name__ == "__main__":

    mp.spawn(launcher, args=(0,), nprocs=2, join=True)
    mp.spawn(launcher, args=(1,), nprocs=2, join=True)
