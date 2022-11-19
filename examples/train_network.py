import numpy_datasets as nds
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch as ch
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, transforms
import numpy as np


def perm(x):
    return x.permute([0, 3, 1, 2])


def launcher(local_rank, option, node_rank=0, gpus=2):
    world_size = gpus * 1
    global_rank = node_rank * gpus + local_rank
    C = 256
    BS = 256
    ch.distributed.init_process_group(
        "nccl",
        rank=global_rank,
        world_size=world_size,
        init_method="tcp://127.0.0.1:23456",
    )
    ch.cuda.set_device(local_rank)

    model = models.resnet18(pretrained=True)
    model.fc = ch.nn.Sequential(
        # ch.nn.Linear(model.fc.in_features, 2048, bias=False),
        # ch.nn.BatchNorm1d(2048),
        # ch.nn.ReLU(True),
        # ch.nn.Linear(2048, 2048, bias=False),
        # ch.nn.BatchNorm1d(2048),
        # ch.nn.ReLU(True),
        ch.nn.Linear(model.fc.in_features, 200),
    )
    model = model.to(f"cuda:{local_rank}")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    train_dataset = nds.loader.H5Dataset(
        "../../DATASETS/TINY_IMAGENET/train.hdf5",
        device=f"cuda:{local_rank}",
        chunkit=C,
        shuffle=True,
        transform=transforms.Compose(
            [
                transforms.Normalize(
                    np.array([0.485, 0.456, 0.406]),
                    np.array([0.229, 0.224, 0.225]),
                ),
            ]
        ),
    )

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BS // C,
        num_workers=10,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
    )

    val_dataset = nds.loader.H5Dataset(
        "../../DATASETS/TINY_IMAGENET/val.hdf5",
        device=f"cuda:{local_rank}",
        chunkit=1,
        shuffle=False,
        transform=transforms.Compose(
            [
                transforms.Normalize(
                    np.array([0.485, 0.456, 0.406]),
                    np.array([0.229, 0.224, 0.225]),
                ),
            ]
        ),
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BS,
        num_workers=10,
        sampler=val_sampler,
        persistent_workers=True,
        shuffle=False,
    )

    optimizer = ch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(130):
        model.eval()
        with ch.no_grad():
            accus = []
            for (x, y) in tqdm(nds.loader.FastLoader(val_loader)):
                # pred = model(x).ge(0).int()
                # c = (2 ** ch.arange(8, device=pred.device) * pred).sum(1)
                # accus.append(ch.eq(c, y).float().mean().item() * 100)
                pred = model(x).argmax(1)
                accus.append(ch.eq(pred, y).float().mean().item() * 100)
            print(np.mean(accus))
        sampler.set_epoch(e)
        model.train()
        for (x, y) in tqdm(nds.loader.FastLoader(train_loader)):
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            # loss = ch.nn.functional.binary_cross_entropy_with_logits(
            #     output, nds.utils.base_two(y, 8).float()
            # )
            loss = ch.nn.functional.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
        print(loss.item())


if __name__ == "__main__":

    mp.spawn(launcher, args=(0,), nprocs=2, join=True)
