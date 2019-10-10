from ssbr.datasets.ircad import IrcadDataset, IrcadData
from ssbr.datasets.utils import DicomVolumeStore, stack_sampler, SSBRDataset
from ssbr.datasets.ops import grey2rgb, resize, rescale, image2np
from pathlib import Path
import numpy as np
from typing import Tuple
import os
import h5py
from ssbr.model import ssbr_model
from keras.callbacks import ModelCheckpoint
from dataclasses import dataclass

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


@dataclass
class TrainConfig:

    # Data config
    resize: Tuple[float, float] = (64, 64)
    window: Tuple[float, float] = (-300, 700)

    # Training config
    lr: float
    batch_size: int
    num_slices: int
    loss_alpha: float


def train_experiment(dataset, config, output):

    volume_transforms = [
        resize((64, 64)),
        image2np,
        rescale(low=-300, high=700, scale=255, dtype=np.uint8),
        grey2rgb,
    ]

    BATCH_SIZE = 4
    NUM_SLICES = 8

    ### DATASET
    FOLDER = Path('./data')
    ircad = IrcadData(FOLDER)
    cache = h5py.File(str(FOLDER / 'ircad.h5'), 'a')
    volumes = DicomVolumeStore(ircad, transforms=volume_transforms, cache=cache)
    dataset = SSBRDataset(volumes=volumes, split=0.2)
    datagen_train = dataset.train(batch_size=BATCH_SIZE, num_slices=NUM_SLICES)
    datagen_valid = dataset.valid(batch_size=BATCH_SIZE, num_slices=NUM_SLICES)

    model_filepath = './data/model.h5'

    ### TRAINING
    config = {}
    m, score_extractor = ssbr_model(lr=config.get('lr', 0.0001),
                                    batch_size=BATCH_SIZE,
                                    num_slices=NUM_SLICES,
                                    alpha=config.get('alpha', 0.5))

    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    mcp = ModelCheckpoint(filepath=model_filepath, monitor='val_loss', verbose=1, save_best_only=True)

    # # Train model
    hist = m.fit_generator(generator=datagen_train,
                           steps_per_epoch=30,
                           epochs=10,
                           callbacks=[mcp],
                           validation_data=datagen_valid,
                           validation_steps=20)
