import os
from pathlib import Path

user = "YOU"
BRATS_TRAIN_FOLDERS = f"/work/a9778875/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
BRATS_VAL_FOLDER = f"/work/a9778875/RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
BRATS_TEST_FOLDER = f"/work/a9778875/trainval/"


def get_brats_folder(on="val"):
    if on == "train":
        return os.environ['BRATS_FOLDERS'] if 'BRATS_FOLDERS' in os.environ else BRATS_TRAIN_FOLDERS
    elif on == "val":
        return os.environ['BRATS_VAL_FOLDER'] if 'BRATS_VAL_FOLDER' in os.environ else BRATS_VAL_FOLDER
    elif on == "test":
        return os.environ['BRATS_TEST_FOLDER'] if 'BRATS_TEST_FOLDER' in os.environ else BRATS_TEST_FOLDER


import torch
print(torch.__version__)


