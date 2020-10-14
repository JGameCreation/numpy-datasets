from tqdm import tqdm
import matplotlib.image as mpimg
import zipfile
import numpy as np
import os
import time
import io
from typing import Any, Callable, List, Iterable, Optional, TypeVar

_CITATION = """\
@inproceedings{conf/iccv/LiuLWT15,
  added-at = {2018-10-09T00:00:00.000+0200},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  biburl = {https://www.bibsonomy.org/bibtex/250e4959be61db325d2f02c1d8cd7bfbb/dblp},
  booktitle = {ICCV},
  crossref = {conf/iccv/2015},
  ee = {http://doi.ieeecomputersociety.org/10.1109/ICCV.2015.425},
  interhash = {3f735aaa11957e73914bbe2ca9d5e702},
  intrahash = {50e4959be61db325d2f02c1d8cd7bfbb},
  isbn = {978-1-4673-8391-2},
  keywords = {dblp},
  pages = {3730-3738},
  publisher = {IEEE Computer Society},
  timestamp = {2018-10-11T11:43:28.000+0200},
  title = {Deep Learning Face Attributes in the Wild.},
  url = {http://dblp.uni-trier.de/db/conf/iccv/iccv2015.html#LiuLWT15},
  year = 2015
}
"""


def download(path: str) -> None:
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path, "celeba-dataset.zip")):
        cwd = os.getcwd()
        os.chdir(path)
        os.system("kaggle datasets download -d jessicali9530/celeba-dataset")
        os.chdir(cwd)


def load(path=None):
    """face images with attributes
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
     with more than 200K celebrity images, each with 40 attribute annotations. The
    images in this dataset cover large pose variations and background clutter.
    CelebA has large diversities, large quantities, and rich annotations, including
    - 10,177 number of identities,
    - 202,599 number of face images, and
    - 5 landmark locations, 40 binary attributes annotations per image.
    The dataset can be employed as the training and test sets for the following
    computer vision tasks: face attribute recognition, face detection, and landmark
     (or facial part) localization.
    Note: CelebA dataset may contain potential bias. The fairness indicators
    `https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb`
    goes into detail about several considerations to keep in mind while using the
    CelebA dataset.

    Parameters
    ----------

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

    download(os.path.join(path, "celebA"))

    t0 = time.time()

    archive = zipfile.ZipFile(os.path.join(path, "celebA", "celeba-dataset.zip"), "r")
    images = []
    ids = []
    for name in tqdm(archive.namelist()):
        if "jpg" in name:
            images.append(mpimg.imread(archive.open(name), "jpg"))

    atts = np.loadtxt(
        archive.open("list_attr_celeba.csv"),
        delimiter=",",
        dtype=str,
    )
    names = atts[0, 1:]

    # list_bbox_celeba.csv
    # list_eval_partition.csv
    # list_landmarks_align_celeba.csv

    print("Dataset celebA loaded in {0:.2f}s.".format(time.time() - t0))
    dataset = {
        "images": np.array(images),
        "attributes": atts[1:, 1:].astype("float32"),
        "names": names,
    }
    return dataset
