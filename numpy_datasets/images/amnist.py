import os
import numpy as np
from ..utils import download_dataset
import time
import io
import matplotlib.image as mpimg
from zipfile import ZipFile


_source = "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset"

cite = """
@article{el2017arabic,
  title={Arabic handwritten characters recognition using convolutional neural network},
  author={El-Sawy, Ahmed and Loey, Mohamed and El-Bakry, Hazem},
  journal={WSEAS Transactions on Computer Research},
  volume={5},
  pages={11--19},
  year={2017}
}"""

_name = "amnist"

_urls = {
    "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Test%20Images%203360x32x32.zip": "TestImages32x32.zip",
    "https://github.com/mloey/Arabic-Handwritten-Characters-Dataset/raw/master/Train%20Images%2013440x32x32.zip": "TrainImages32x32.zip",
}


def load(path=None):
    """Arabic Handwritten Characters Dataset

    Astract
    Handwritten Arabic character recognition systems face several challenges, including the unlimited variation in human handwriting and large public databases. In this work, we model a deep learning architecture that can be effectively apply to recognizing Arabic handwritten characters. A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in supervised mode. The CNN trained and tested our database that contain 16800 of handwritten Arabic characters. In this paper, the optimization methods implemented to increase the performance of CNN. Common machine learning methods usually apply a combination of feature extractor and trainable classifier. The use of CNN leads to significant improvements across different machine-learning classification algorithms. Our proposed CNN is giving an average 5.1% misclassification error on testing data.

    Context
    The motivation of this study is to use cross knowledge learned from multiple works to enhancement the performance of Arabic handwritten character recognition. In recent years, Arabic handwritten characters recognition with different handwriting styles as well, making it important to find and work on a new and advanced solution for handwriting recognition. A deep learning systems needs a huge number of data (images) to be able to make a good decisions.

    Content
    The data-set is composed of 16,800 characters written by 60 participants, the age range is between 19 to 40 years, and 90% of participants are right-hand. Each participant wrote each character (from ’alef’ to ’yeh’) ten times on two forms as shown in Fig. 7(a) & 7(b). The forms were scanned at the resolution of 300 dpi. Each block is segmented automatically using Matlab 2016a to determining the coordinates for each block. The database is partitioned into two sets: a training set (13,440 characters to 480 images per class) and a test set (3,360 characters to 120 images per class). Writers of training set and test set are exclusive. Ordering of including writers to test set are randomized to make sure that writers of test set are not from a single institution (to ensure variability of the test set).


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

    t0 = time.time()
    download_dataset(path, _name, _urls)
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    with ZipFile(os.path.join(path, _name, "TestImages32x32.zip")) as archive:
        for entry in archive.infolist():
            print(entry.filename)
            if ".png" not in entry.filename:
                continue
            content = archive.read(entry)
            test_images.append(mpimg.imread(io.BytesIO(content), "png"))
            test_labels.append(int(entry.filename.split("_")[-1][:-4]))

    with ZipFile(os.path.join(path, _name, "TrainImages32x32.zip")) as archive:
        for entry in archive.infolist():
            print(entry.filename)
            if ".png" not in entry.filename:
                continue
            content = archive.read(entry)
            train_images.append(mpimg.imread(io.BytesIO(content), "png"))
            train_labels.append(int(entry.filename.split("_")[-1][:-4]))

    data = {
        "train_set/images": np.array(train_images),
        "train_set/labels": np.array(train_labels),
        "test_set/images": np.array(test_images),
        "test_set/labels": np.array(test_labels),
    }

    print("Dataset kmnist loaded in {0:.2f}s.".format(time.time() - t0))

    return data
