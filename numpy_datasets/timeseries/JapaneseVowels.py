import os
from ..utils import download_dataset, load_from_tsfile_to_dataframe
import pathlib


def load(path=None):
    """See http://www.timeseriesclassification.com/description.php?Dataset=JapaneseVowels
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

    path = pathlib.Path(path) / "JapaneseVowels"
    download_dataset(
        path,
        {
            "JapaneseVowels.zip": "http://www.timeseriesclassification.com/Downloads/JapaneseVowels.zip"
        },
        extract=True,
    )

    path = path / "extracted_JapaneseVowels"

    X_train, y_train = load_from_tsfile_to_dataframe(
        path / "JapaneseVowels/JapaneseVowels_TRAIN.ts"
    )
    X_test, y_test = load_from_tsfile_to_dataframe(
        path / "JapaneseVowels/JapaneseVowels_TEST.ts"
    )
    return (X_train, y_train), (X_test, y_test)
