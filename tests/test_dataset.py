from ssbr.datasets.utils import DicomVolumeStore, batcher, cyclic_shuffler, slice_sampler
from unittest.mock import patch, Mock
import numpy as np

import h5py


def test_batcher():
    a = [1, 2, 3, 4, 5]
    b = batcher(a, 3)

    batch1 = next(b)
    batch2 = next(b)

    assert all([aa == bb for aa, bb in zip(batch1, [1, 2, 3])])
    assert all([aa == bb for aa, bb in zip(batch2, [4, 5])])


def test_shuffler():

    items = [1, 2, 3, 4, 5]

    it = cyclic_shuffler(items)
    for i in range(10):
        lst = []
        for j in range(5):
            lst.append(next(it))
        lst.sort()

        # Assert list contains all items at each cycle
        assert all([aa == bb for aa, bb in zip(lst, items)])


def test_slice_sampler():
    vol = np.random.rand(30, 64, 64)

    stack = slice_sampler(vol, num_slices=6)
    assert stack.shape == (6, 64, 64)


@patch('ssbr.datasets.utils.load_dicom')
def test_dicom_store(load_dicom_mock: Mock, tmpdir):
    dcm_map = {
        'dcm1': np.random.rand(20, 64, 64),
        'dcm2': np.random.rand(20, 64, 64),
        'dcm3': np.random.rand(20, 64, 64),
    }

    load_dicom_mock.return_value = np.random.rand(20, 64, 64)

    def set_zero(x):
        x[:] = 0.0
        return x

    tmpfile = tmpdir / 'test.h5'
    tmpcache = h5py.File(str(tmpfile), 'w')
    store = DicomVolumeStore(volumes=dcm_map, transforms=[set_zero], cache=tmpcache)

    for vid in store:
        vol = np.asarray(store[vid])
        assert vol.shape == (20, 64, 64)
        assert vid in tmpcache
        np.testing.assert_equal(np.zeros((20, 64, 64)), vol)
