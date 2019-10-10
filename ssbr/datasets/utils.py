from typing import List, Callable, Mapping, Tuple, Iterable, Iterator
import h5py
import numpy as np
import SimpleITK as sitk
from ssbr.datasets.ops import compose, load_dicom
from itertools import chain, islice
import random
from torch.utils.data import IterableDataset


class SSBRDataset(IterableDataset):
    def __init__(self, volumes: Mapping, split=0.2):
        self.volumes = volumes
        num_valid = int(len(volumes) // (1 / split))
        vids = list(volumes.keys())
        self._valid_vids = random.choices(vids, k=num_valid)
        self._train_vids = list(set(vids) - set(self._valid_vids))

    def train(self, batch_size=5, num_slices=8):
        stacks_generator = stack_sampler(self.volumes,
                                         volume_ids=self._train_vids,
                                         batch_size=batch_size,
                                         num_slices=num_slices)
        while True:
            yield (stacks_generator.__next__(), np.zeros((batch_size, num_slices, 1)))

    def valid(self, batch_size=5, num_slices=8):
        stacks_generator = stack_sampler(self.volumes,
                                         volume_ids=self._valid_vids,
                                         batch_size=batch_size,
                                         num_slices=num_slices)
        while True:
            yield (stacks_generator.__next__(), np.zeros((batch_size, num_slices, 1)))

    def evaluate(self, batch_size=5):
        raise NotImplementedError


class DicomVolumeStore(Mapping):
    """
    Mapping between volume ID and data. 
    Optionally use cache file.
    Apply volume transformation volume transformations
    """
    def __init__(self,
                 volumes: Mapping[str, str],
                 transforms: List[Callable] = None,
                 cache: Mapping[str, np.ndarray] = None):
        """
        :param volumes: Mapping between volume ID and dicom folder path
        :param transforms: List of volume-wise transformations
        :param cache: Cache system to prioritize over loading dicoms
        """
        self.volumes = volumes
        self.pipeline = compose(transforms)
        self.cache = cache

    def __len__(self):
        return len(self.volumes)

    def __iter__(self):
        yield from self.volumes

    def __getitem__(self, key) -> np.ndarray:

        # Handle key object
        if self.cache:
            if key not in self.cache:
                # Load dicom volume
                vol = load_dicom(self.volumes[key])

                # Apply transforms
                tvol = next(self.pipeline(iter([vol])))

                chunk_size = (1, ) + tvol.shape[1:]
                self.cache.create_dataset(name=key, data=tvol, chunks=chunk_size, compression='lzf')
            return self.cache[key]
        else:
            vol = load_dicom(self.volumes[key])
            tvol = next(self.pipeline(iter([vol])))
            return tvol

    def build_cache(self):
        """Load every cases to build cache"""
        for vid in self.volumes:
            arr = self[vid]
            print(f'Building cache for {vid} - {arr.shape}')


def slice_sampler(volume: np.ndarray, num_slices: int = 8, equidistance_range: Tuple[int] = None):
    """
    Sample equidistant slices inside a volume. The distance between each slice is randomly selected
    within `equidistance_range`.
    """

    if equidistance_range is None:
        equidistance_range = (1, 4)

    shp = volume.shape
    equi_dist = random.randint(*equidistance_range)

    # Find span of sample slices
    span = equi_dist * num_slices
    min_starting_slice = 0
    max_starting_slice = shp[0] - span
    starting_slice = random.randint(min_starting_slice, max_starting_slice)

    stack = []
    for i in range(num_slices):
        sampling_slice = starting_slice + i * equi_dist
        I = np.asarray(volume[sampling_slice])
        stack.append(I)

    return np.asarray(stack)


def batcher(iterable: Iterable, size: int):
    """Generate batches of items from an iterable"""
    sourceiter = iter(iterable)
    while True:
        try:
            batchiter = islice(sourceiter, size)
            batch = chain([next(batchiter)], batchiter)
            yield np.asarray([it for it in batch])
        except StopIteration:
            return


def cyclic_shuffler(items: Iterable) -> Iterator:
    """Infinitely cycles through a list of items while shuffling the list at each cycle"""
    while True:
        shuffled_list = list(items)
        random.shuffle(shuffled_list)
        for element in shuffled_list:
            yield element


def stack_sampler(volumes: DicomVolumeStore, volume_ids=None, batch_size=5, num_slices=8):
    """Sample a random batch of slice stacks"""

    if not volume_ids:
        volume_ids = list(volumes.keys())

    # Shuffle volume ids
    shuffled_ids = cyclic_shuffler(volume_ids)

    # Get the volume for each volume ID
    volume_data = (volumes[vid] for vid in shuffled_ids)

    # Sample a stack of slices for each volume
    slice_stack = (slice_sampler(vol, num_slices) for vol in volume_data)

    # Batch slice stack together
    stack_batcher = batcher(slice_stack, batch_size)

    return stack_batcher
