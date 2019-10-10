import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from keras.callbacks import ModelCheckpoint

from ssbr.datasets.ircad import IrcadData
from ssbr.datasets.ops import grey2rgb, image2np, rescale, resize
from ssbr.datasets.utils import DicomVolumeStore, SSBRDataset, stack_sampler
from ssbr.model import ssbr_model
from ssbr import DATAFOLDER

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# DATAFOLDER = Path('./data')


@dataclass
class TrainConfig:

    # Data config
    resize: Tuple[float, float] = (64, 64)
    window: Tuple[float, float] = (-300, 700)
    train_valid_split: float = 0.2
    equidistance_range: Tuple[int, int] = (1, 6)

    # Training config
    lr: float = 0.0001
    batch_size: int = 5
    num_slices: int = 8
    loss_alpha: float = 0.5

    # Experiment config
    num_epochs: int = 50
    steps_per_epoch: int = 30
    valid_steps: int = 20

    def __post_init__(self):
        # Required since it could be loaded from .json which don't support tuple
        self.resize = tuple(self.resize)


def train_experiment(config, dataset, output):

    ### CONFIG
    if not isinstance(config, TrainConfig):
        config = TrainConfig(**config)

    # Build output
    output = Path(output)
    os.makedirs(output, exist_ok=True)

    # Save config
    complete_config = asdict(config)
    config_fp = output / 'config.json'
    with open(config_fp, 'w') as fid:
        json.dump(complete_config, fid, indent=4)

    ### DATASET
    if dataset == 'ircad':
        volume_folder = DATAFOLDER / 'ircad'
        volume_files = IrcadData(volume_folder)
    else:
        raise NotImplementedError(f'Unknown dataset {dataset}')

    # Volume transformation pipeline
    volume_transforms = [
        resize(config.resize),
        image2np,
        rescale(low=config.window[0], high=config.window[1], scale=255, dtype=np.uint8),
        grey2rgb,
    ]

    cache = h5py.File(str(volume_folder / 'cache.h5'), 'a')
    volumes = DicomVolumeStore(volume_files, transforms=volume_transforms, cache=cache)
    dataset = SSBRDataset(volumes=volumes, split=config.train_valid_split)

    datagen_train = dataset.train(batch_size=config.batch_size,
                                  num_slices=config.num_slices,
                                  equidistance_range=config.equidistance_range)

    datagen_valid = dataset.valid(batch_size=config.batch_size,
                                  num_slices=config.num_slices,
                                  equidistance_range=config.equidistance_range)

    # Save split
    split_fp = output / 'split.json'
    split = {'train': dataset._train_vids, 'valid': dataset._valid_vids}
    with open(split_fp, 'w') as fid:
        json.dump(split, fid)

    ### TRAINING
    model_filepath = output / 'model.h5'
    m, score_extractor = ssbr_model(lr=config.lr,
                                    batch_size=config.batch_size,
                                    num_slices=config.num_slices,
                                    alpha=config.loss_alpha)

    mcp = ModelCheckpoint(filepath=str(model_filepath), monitor='val_loss', verbose=1, save_best_only=True)

    hist = m.fit_generator(generator=datagen_train,
                           steps_per_epoch=config.steps_per_epoch,
                           epochs=config.num_epochs,
                           callbacks=[mcp],
                           validation_data=datagen_valid,
                           validation_steps=config.valid_steps)

    # Keras history has np.float32 which are not json serializable
    class HistoryEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.floating):
                return float(obj)

    history_filepath = output / 'history.json'
    with open(history_filepath, 'w') as f:
        json.dump(hist.history, f, indent=4, cls=HistoryEncoder)