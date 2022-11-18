import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch as ch


def mapit(batch):
    return [b.to(device="cuda", non_blocking=True) for b in batch]


    
def fn(batch):
    elem = batch[0][0]
    N = sum([b[0].size(0) for b in batch])
    out = None
    print([b[0].device for b in batch])
    if ch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = elem.numel() * N
        out = elem.new_empty(numel, device="cuda").resize_(N, *list(elem.shape))
        return (
            out.copy_(ch.stack([b[0] for b in batch]), non_blocking=True),
            0,
        )
    return ch.stack([b[0] for b in batch]), 0


def collate_fn(device):
    ch.multiprocessing.set_start_method("spawn")  # good solution !!!!
    return fn


def dataset_to_h5(dataset, h5file, num_workers=16, chunk_size=1024):

    nfiles = len(dataset)

    def collate_fn(images):
        data = np.empty(
            (len(images), images[0][0].height, images[0][0].width, 3), dtype="uint8"
        )
        labels = np.empty((len(images),), dtype="uint")
        for i, (image, label) in enumerate(images):
            # we save the byte format (compressed)
            data[i] = np.asarray(image)
            labels[i] = label
        return data, labels

    loader = DataLoader(
        dataset,
        batch_size=chunk_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    with h5py.File(h5file, "w") as h5f:
        label_ds = h5f.create_dataset("labels", shape=(nfiles,), dtype=int)
        h5f.create_dataset("chunk_size", data=[chunk_size])
        for i, (x, y) in tqdm(enumerate(loader), total=len(loader), desc="converting"):
            h5f.create_dataset(f"images_{i}", data=x)
            label_ds[i * num_workers : i * num_workers + len(y)] = y


class H5Dataset(Dataset):
    def __init__(
        self,
        hdf5file,
        imgs_key="images",
        labels_key="labels",
        transform=None,
        device="cpu",
        chunkit=1,
    ):

        self.chunkit = chunkit
        self.hdf5file = hdf5file
        self.device = device
        self.imgs_key = imgs_key
        self.labels_key = labels_key
        self.transform = transform
        # return len(self.db[self.labels_key])
        with h5py.File(self.hdf5file, "r") as db:
            self.lens = len(db[labels_key]) // chunkit
            self.datasets = [i for i in db.keys() if imgs_key in i]
            self.datasets.sort()
            self.chunk_size = db["chunk_size"][0]

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        chunk = idx // self.chunk_size
        index = idx - chunk * self.chunk_size
        extra = np.random.choice(
            range(1, self.chunk_size), size=self.chunkit - 1, replace=False
        )
        indices = np.concatenate([[index], index + extra]) % self.chunk_size
        indices = np.sort(indices)
        with h5py.File(self.hdf5file, "r") as db:
            label = db[self.labels_key][indices + chunk * self.chunk_size]
            image = db[f"{self.imgs_key}_{chunk}"][indices, :, :, :]
            label = ch.from_numpy(label)
            image = ch.from_numpy(image)
        if self.transform:
            image = self.transform(image)
        return image, label
