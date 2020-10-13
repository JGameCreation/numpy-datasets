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
@inproceedings{gourier2004estimating,
  title={Estimating face orientation from robust detection of salient facial features},
  author={Gourier, Nicolas and Hall, Daniela and Crowley, James L},
  booktitle={ICPR International Workshop on Visual Observation of Deictic Gestures},
  year={2004},
  organization={Citeseer}
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
    """head angle classification
    The head pose database consists of 15 sets of images. Each set contains of 2 series of 93 images of the same person at different poses. There are 15 people in the database, wearing glasses or not and having various skin color. The pose, or head orientation is determined by 2 angles (h,v), which varies from -90 degrees to +90 degrees. Here is a sample of a serie :

     PersonID = {01, ..., 15}:
                stands for the number of the person.

    Serie =  {1, 2}
                stands for the number of the serie.

    Number = {00, 01, ..., 92}
                the number of the file in the directory.

    VerticalAngle = {-90, -60, -30, -15, 0, +15, +30, +60, +90}

    HorizontalAngle = {-90, -75, -60, -45, -30, -15, 0, +15, +30, +45, +60, +75, +90}

    All images have been taken using the FAME Platform of the PRIMA Team in INRIA Rhone-Alpes. To obtain different poses, we have put markers in the whole room. Each marker corresponds to a pose (h,v). Post-it are used as markers. The whole set of post-it covers a half-sphere in front of the person.

    In order to obtain the face in the center of the image, the person is asked to adjust the chair to see the device in front of him. After this initialization phase, we ask the person to stare successively at 93 post-it notes, without moving his eyes. This second phase just takes a few minutes.

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
