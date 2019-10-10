import logging
import os
import random
import threading
import zipfile
from collections.abc import Mapping
from pathlib import Path
from urllib import request

import h5py
import numpy as np
from torch.utils.data import IterableDataset
from tqdm import tqdm

from ssbr.datasets.utils import DicomVolumeStore, stack_sampler

IRCAD_DATASET_URL = "https://www.ircad.fr/softwares/3Dircadb/3Dircadb1/3Dircadb1.zip"
IRCAD_ARCHIVE_NAME = '3Dircadb1.zip'
IRCAD_DATA_FOLDER = 'data'
IRCAD_DICOM_FOLDER = 'PATIENT_DICOM'

LOG = logging.getLogger('ssbr.dataset.ircad')


class DowloadProgress(tqdm):
    """Progress bar for urllib's urlretrieve reportook"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def maybe_download(folder: Path):
    """
    Download the IRCAD zip archive at the provided location if the file is do not already exists
    """
    fp = folder / IRCAD_ARCHIVE_NAME
    if not fp.exists():
        LOG.info(f'Downloading ircad dataset from url {IRCAD_DATASET_URL}')
        with DowloadProgress(unit='B',
                             unit_scale=True,
                             miniters=1,
                             desc=IRCAD_ARCHIVE_NAME) as t:
            request.urlretrieve(IRCAD_DATASET_URL,
                                filename=fp,
                                reporthook=t.update_to)
    else:
        LOG.info(f'IRCAD dataset already downloaded at {IRCAD_DATASET_URL}')


def maybe_unzip(folder: Path) -> Path:
    """Recursively unzip ircad dicom folders if not already done"""
    archive_fp = folder / IRCAD_ARCHIVE_NAME
    data_folder = folder / IRCAD_DATA_FOLDER
    if not data_folder.exists():
        LOG.info(f'Extracting ircad archive to {IRCAD_DATA_FOLDER}')
        with zipfile.ZipFile(archive_fp, 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    # Extract nested dicom folders for each cases
    folders = [x for x in data_folder.iterdir() if x.is_dir()]
    for folder in folders:
        dcm_folder = folder / IRCAD_DICOM_FOLDER
        dcm_archive = folder / (IRCAD_DICOM_FOLDER + '.zip')

        if not dcm_folder.exists():
            LOG.info(f'Extracting dicom archive for patient {folder}')
            with zipfile.ZipFile(dcm_archive, 'r') as zip_ref:
                zip_ref.extractall(folder)


class IrcadData(Mapping):
    """
    Download and unzip data if needed
    """
    def __init__(self, root):
        self.root = Path(root)
        os.makedirs(self.root, exist_ok=True)

        maybe_download(self.root)
        maybe_unzip(self.root)

        data_dir = self.root / IRCAD_DATA_FOLDER
        self.volumes = {
            x.name: str(x / IRCAD_DICOM_FOLDER)
            for x in data_dir.iterdir() if x.is_dir()
        }

    def __iter__(self):
        yield from self.volumes

    def __getitem__(self, key):
        return self.volumes[key]

    def __len__(self):
        return len(self.volumes)