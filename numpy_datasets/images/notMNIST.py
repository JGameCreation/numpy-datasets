import time
import os, tarfile, io
import numpy as np
from ..utils import download_dataset


_description_url = "https://zenodo.org/record/259444"

_name = "HASYv2"
cite = """@misc{thoma, martin_2017, title={HASYv2 - Handwritten Symbol database}, DOI={10.5281/zenodo.259444}, abstractNote={<p>HASY contains 32px x 32px images of 369 symbol classes. In total, HASY contains over 150,000 instances of handwritten symbols.<br> <br> See "The HASYv2 dataset" paper (https://arxiv.org/abs/1701.08380) for more information.</p>}, publisher={Zenodo}, author={Thoma, Martin}, year={2017}, month={Jan}}"""

_urls = {
    "http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz": "otMNIST_large.tar.gz",
}


def load(path=None):
    """32x32 B&W handwritten symbol classification.

    HASY contains 32px x 32px images of 369 symbol classes. In total, HASY
    contains over 150,000 instances of handwritten symbols.

    The  HASYv2  dataset  contains  369  classes.  Those  classesinclude  the  Latin  uppercase  and  lowercase  characters  (A-Z,a-z), the Arabic numerals (0-9), 32 different types of arrows,fractal  and  calligraphic  Latin  characters,  brackets  and  more.See Tables VI to XIV for more information.

    The HASYv2 dataset contains168 233black and white imagesof the size32px×32px. Each image is labeled with one of369 labels. An example of 100 elements of the HASYv2 dataset is shown in Figure 1.The average amount of black pixels is16%, but this is highlyclass-dependent ranging from 3.7% of “...” to 59.2% of “”average black pixel by class.


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
    file_ = tarfile.open(os.path.join(path, _name, "HASYv2.tar.bz2"), "r:gz")
    # loading test label
    read_file = file_.open("stl10_binary/test_y.bin").read()
    test_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8) - 1
    # loading train label
    read_file = file_.extractfile("stl10_binary/train_y.bin").read()
    train_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8) - 1
    # load test images
    read_file = file_.extractfile("stl10_binary/test_X.bin").read()
    test_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )
    # load train images
    read_file = file_.extractfile("stl10_binary/train_X.bin").read()
    train_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )
    # load unlabelled images
    read_file = file_.extractfile("stl10_binary/unlabeled_X.bin").read()
    unlabeled_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )

    print("Dataset stl10 loaded in", "{0:.2f}".format(time.time() - t), "s.")
    data = {
        "train_set/images": train_X,
        "train_set/labels": train_y,
        "test_set/images": test_X,
        "test_set/labels": test_y,
        "unlabelled": unlabeled_X,
    }
    return data
