import os
import numpy as np
from ..utils import download_dataset
import time

_source = "https://github.com/rois-codh/kmnist/blob/master/README.md"

cite = """
@online{clanuwat2018deep,
  author       = {Tarin Clanuwat and Mikel Bober-Irizar and Asanobu Kitamoto and Alex Lamb and Kazuaki Yamamoto and David Ha},
  title        = {Deep Learning for Classical Japanese Literature},
  date         = {2018-12-03},
  year         = {2018},
  eprintclass  = {cs.CV},
  eprinttype   = {arXiv},
  eprint       = {cs.CV/1812.01718},
}"""

_dataset = "kmnist"

_urls_kmnist = {
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-imgs.npz": "kmnist-train-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-train-labels.npz": "kmnist-train-labels.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-imgs.npz": "kmnist-test-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/kmnist/kmnist-test-labels.npz": "kmnist-test-labels.npz",
}

_urls_k49mnist = {
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz": "k49-train-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz": "k49-train-labels.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz": "k49-test-imgs.npz",
    "http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz": "k49-test-labels.npz",
}


def load(dataset="kmnist", path=None):
    """japanese character (image) classification

    Kuzushiji-MNIST is a drop-in replacement for the MNIST dataset (28x28 grayscale, 70,000 images), provided in the original MNIST format as well as a NumPy format. Since MNIST restricts us to 10 classes, we chose one character to represent each of the 10 rows of Hiragana when creating Kuzushiji-MNIST.

    Kuzushiji-49, as the name suggests, has 49 classes (28x28 grayscale, 270,912 images), is a much larger, but imbalanced dataset containing 48 Hiragana characters and one Hiragana iteration mark.

    Kuzushiji-MNIST

    Kuzushiji-MNIST contains 70,000 28x28 grayscale images spanning 10 classes (one from each column of hiragana), and is perfectly balanced like the original MNIST dataset (6k/1k train/test for each class).
    File    Examples    Download (MNIST format)     Download (NumPy format)
    Training images     60,000  train-images-idx3-ubyte.gz (18MB)   kmnist-train-imgs.npz (18MB)
    Training labels     60,000  train-labels-idx1-ubyte.gz (30KB)   kmnist-train-labels.npz (30KB)
    Testing images  10,000  t10k-images-idx3-ubyte.gz (3MB)     kmnist-test-imgs.npz (3MB)
    Testing labels  10,000  t10k-labels-idx1-ubyte.gz (5KB)     kmnist-test-labels.npz (5KB)

    Mapping from class indices to characters: kmnist_classmap.csv (1KB)

    We recommend using standard top-1 accuracy on the test set for evaluating on Kuzushiji-MNIST.
    Which format do I download?

    If you're looking for a drop-in replacement for the MNIST or Fashion-MNIST dataset (for tools that currently work with these datasets), download the data in MNIST format.

    Otherwise, it's recommended to download in NumPy format, which can be loaded into an array as easy as:
    arr = np.load(filename)['arr_0'].
    Kuzushiji-49

    Kuzushiji-49 contains 270,912 images spanning 49 classes, and is an extension of the Kuzushiji-MNIST dataset.
    File    Examples    Download (NumPy format)
    Training images     232,365     k49-train-imgs.npz (63MB)
    Training labels     232,365     k49-train-labels.npz (200KB)
    Testing images  38,547  k49-test-imgs.npz (11MB)
    Testing labels  38,547  k49-test-labels.npz (50KB)

    Mapping from class indices to characters: k49_classmap.csv (1KB)

    We recommend using balanced accuracy on the test set for evaluating on Kuzushiji-49.
    We use the following implementation of balanced accuracy:

    License

    Both the dataset itself and the contents of this repo are licensed under a permissive CC BY-SA 4.0 license, except where specified within some benchmark scripts. CC BY-SA 4.0 license requires attribution, and we would suggest to use the following attribution to the KMNIST dataset.

    "KMNIST Dataset" (created by CODH), adapted from "Kuzushiji Dataset" (created by NIJL and others), doi:10.20676/00000341

    Parameters
    ----------

        dataset: str(optional)
            `"kmnist"` or `"k49mnist"`
        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        valid_images: array

        valid_labels: array

        test_images: array

        test_labels: array

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    t0 = time.time()
    if dataset == "kmnist":
        download_dataset(path, _dataset, _urls_kmnist)
        train_imgs = np.load(os.path.join(path, _dataset, "kmnist-train-imgs.npz"))[
            "arr_0"
        ]
        test_imgs = np.load(os.path.join(path, _dataset, "kmnist-test-imgs.npz"))[
            "arr_0"
        ]
        train_labels = np.load(os.path.join(path, _dataset, "kmnist-train-labels.npz"))[
            "arr_0"
        ]
        test_labels = np.load(os.path.join(path, _dataset, "kmnist-test-labels.npz"))[
            "arr_0"
        ]
    else:
        download_dataset(path, _dataset, _urls_k49mnist)
        train_imgs = np.load(os.path.join(path, _dataset, "k49-train-imgs.npz"))[
            "arr_0"
        ]
        test_imgs = np.load(os.path.join(path, _dataset, "k49-test-imgs.npz"))["arr_0"]
        train_labels = np.load(os.path.join(path, _dataset, "k49-train-labels.npz"))[
            "arr_0"
        ]
        test_labels = np.load(os.path.join(path, _dataset, "k49-test-labels.npz"))[
            "arr_0"
        ]

    data = {
        "train_set/images": train_imgs,
        "train_set/labels": train_labels,
        "test_set/images": test_imgs,
        "test_set/labels": test_labels,
    }

    print("Dataset kmnist loaded in {0:.2f}s.".format(time.time() - t0))

    return data
