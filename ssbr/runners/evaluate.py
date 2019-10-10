from ssbr.evaluation import evaluate_experiment
from ssbr.datasets.ircad import IrcadDataset
from ssbr.datasets.ops import grey2rgb, resize, rescale, image2np
from pathlib import Path
import numpy as np

volume_transforms = [
    resize((64, 64)),
    image2np,
    rescale(low=-300, high=700, scale=255, dtype=np.uint8),
    grey2rgb,
]

datagen_train = IrcadDataset(folder=Path('./data'),
                             volume_transforms=volume_transforms,
                             batch_size=BATCH_SIZE,
                             num_slices=NUM_SLICES)

evaluate_experiment(datagen_train, "./data/model.h5", "./data/model1/")
