from tqdm import tqdm
import matplotlib.image as mpimg
import tarfile
import numpy as np
import os
import time
import io
import re
from typing import Any, Callable, List, Iterable, Optional, TypeVar
from ..utils import download_dataset

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

_name = "face_pointing"
_urls = {
    "http://www-prima.inrialpes.fr/perso/Gourier/Faces/Person{:02}-1.tar.gz".format(
        i + 1
    ): "Person{:02}-1.tar.gz".format(i + 1)
    for i in range(15)
}

_urls.update(
    {
        "http://www-prima.inrialpes.fr/perso/Gourier/Faces/Person{:02}-2.tar.gz".format(
            i + 1
        ): "Person{:02}-2.tar.gz".format(i + 1)
        for i in range(15)
    }
)


def load(path=None):
    """
    CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset
     with more than 200K celebrity images, each with 40 attribute annotations. The \
    images in this dataset cover large pose variations and background clutter. \
    CelebA has large diversities, large quantities, and rich annotations, including\
     - 10,177 number of identities,
     - 202,599 number of face images, and
     - 5 landmark locations, 40 binary attributes annotations per image.
    The dataset can be employed as the training and test sets for the following \
    computer vision tasks: face attribute recognition, face detection, and landmark\
     (or facial part) localization.
    Note: CelebA dataset may contain potential bias. The fairness indicators
    [example](https://github.com/tensorflow/fairness-indicators/blob/master/fairness_indicators/documentation/examples/Fairness_Indicators_TFCO_CelebA_Case_Study.ipynb)
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

    download_dataset(path, _name, _urls)

    t0 = time.time()
    images = []
    ids = []
    vert_angles = []
    horiz_angles = []
    for filename in _urls.values():
        with tarfile.open(os.path.join(path, _name, filename), "r:gz") as so:
            for member in so.getmembers():
                ids.append(int(member.name.split("personne")[1][:2]))
                v, h = re.findall("([+-]\d+)", member.name)
                vert_angles.append(int(v))
                horiz_angles.append(int(h))
                f = so.extractfile(member)
                content = f.read()
                images.append(mpimg.imread(io.BytesIO(content), "jpg"))

    print("Dataset {} loaded in {}s.".format(_name, time.time() - t0))
    dataset = {
        "images": np.array(images),
        "vert_angles": np.array(vert_angles),
        "horiz_angles": np.array(horiz_angles),
        "person_ids": np.array(ids),
    }
    return dataset
