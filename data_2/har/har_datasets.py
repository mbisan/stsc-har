import os
from typing import Tuple, List

import numpy as np

# Load datasets predefined

# pylint: disable=invalid-name too-many-arguments too-many-locals

def load_dataset(dataset_dir: str) -> List[List[Tuple[np.ndarray, np.ndarray]]]:

    subject_folders = sorted(
        list(filter(lambda x: "subject" in x, os.listdir(dataset_dir)))
    )

    dataset: List[List[Tuple[np.ndarray, np.ndarray]]] = []

    for subject in subject_folders:

        dataset.append([])

        subject_dir = os.path.join(dataset_dir, subject)

        subject_sts = sorted(
            list(filter(lambda x: "sensor" in x, os.listdir(subject_dir)))
        )

        for sts in subject_sts:
            dataset[-1].append(
                (np.load(os.path.join(subject_dir, sts)),
                 np.load(os.path.join(subject_dir, sts.replace("sensor", "label"))))
            )

    return dataset

load_dataset("./datasets/UCI-HAPT")