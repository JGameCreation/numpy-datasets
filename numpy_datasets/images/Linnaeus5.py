import time
import os, rarfile, io
import numpy as np
from ..utils import download_dataset


_description_url = "http://chaladze.com/l5/"

_name = "Linnaeus5"

_urls = {
    "http://chaladze.com/l5/img/Linnaeus%205%20256X256.rar": "Linnaeus5.rar",
}


def load(path=None):
    """256x256 RGB images classification.


    5 classes: berry, bird, dog, flower, other (negative set)
    Images are 256x256 pixels, color (downsampled versions: 128X128, 64X64 and 32X32 pixels).
    1200 training images, 400 test images per class.
    Images were downloaded from pixabay.com.


    Parameters
    ----------

    path: str (optional)
        the path to look for the data and where it will be downloaded if
        not present

    Returns
    -------

    train_images: array
        the training images

    train_labels: array
        the training labels

    test_images: array
        the test images

    test_labels: array
        the test labels

    extra_images: array
        the unlabeled additional images
    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _name, _urls)

    print("Loading ", _name)
    t = time.time()

    # Loading Dataset
    rar_path = rarfile.RarFile(os.path.join(path, _name, "Linnaeus5.rar"))

    rar_file = rarfile.RarFile.open(rar_path, filename)

    print("Dataset stl10 loaded in", "{0:.2f}".format(time.time() - t), "s.")
    data = {
        "train_set/images": train_X,
        "train_set/labels": train_y,
        "test_set/images": test_X,
        "test_set/labels": test_y,
        "unlabelled": unlabeled_X,
    }
    return data
