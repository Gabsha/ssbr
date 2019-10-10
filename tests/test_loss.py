import keras.backend as K
import numpy as np
import pytest
from keras.layers import Input

from ssbr.loss import HUBER_DELTA, loss_distance, loss_order, smoothL1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def np_smooth_l1(true, pred):
    x = np.abs(pred - true)
    idx_inf = np.where(x < HUBER_DELTA)
    idx_sup = np.where(x >= HUBER_DELTA)
    l = np.zeros(x.shape)
    l[idx_inf] = 0.5 * x[idx_inf] * x[idx_inf]
    l[idx_sup] = HUBER_DELTA * (x[idx_sup] - 0.5 * HUBER_DELTA)
    return l


@pytest.fixture
def loss_order_func():
    num_slices = 8
    inp = Input(shape=(num_slices, ))
    out = loss_order(None, inp)
    return K.function(inputs=[inp], outputs=[out])


@pytest.fixture
def loss_huber_func():
    y_true = K.placeholder(shape=(10, ))
    y_pred = K.placeholder(shape=(10, ))
    out = smoothL1(y_true, y_pred)
    return K.function(inputs=[y_true, y_pred], outputs=[out])


@pytest.fixture
def loss_distance_func():
    num_slices = 8  # Number of slices per volume
    inp_num = Input(shape=(num_slices, ))
    out = loss_distance(None, inp_num)
    return K.function(inputs=[inp_num], outputs=[out])


def test_order_loss(loss_order_func):
    """Test order loss function computes and yield float32 values"""

    # Define a random batch
    x_batch = np.random.rand(10, 8)  # Ten volumes with 8 slices each

    # Assert a single float value is yielded
    loss_rand = loss_order_func(x_batch)
    assert isinstance(loss_rand[0], np.float32)  # Take first output

    # Assert same values are reached with numpy
    x_diff = x_batch[:, 1:] - x_batch[:, 0:-1]
    loss_rand_np = -np.sum(np.log(sigmoid(x_diff))).astype(np.float32)
    np.testing.assert_almost_equal(loss_rand, loss_rand_np, decimal=3)


def test_order_loss_equidistant(loss_order_func):
    """Test order loss function yields small value for equidistant slices"""

    # Assert equally spaced scores give proper values
    a = np.array(range(1, 9))
    x_batch = np.tile(a, [10, 1])
    x_diff = x_batch[:, 1:] - x_batch[:, 0:-1]
    loss_equi = loss_order_func([x_batch])

    loss_equi_np = -np.sum(np.log(sigmoid(x_diff))).astype(np.float32)
    np.testing.assert_almost_equal(loss_equi, loss_equi_np, decimal=3)


def test_huber_loss(loss_huber_func):

    y_true_np = np.random.rand(10, )
    y_pred_np = np.random.rand(10, )

    loss = loss_huber_func([y_true_np, y_pred_np])
    assert isinstance(loss[0], np.float32)

    # Assert with numpy
    loss_np = np.sum(np_smooth_l1(y_true_np, y_pred_np))
    np.testing.assert_almost_equal(loss, loss_np, decimal=3)


def test_distance_loss(loss_distance_func):

    # Define sample batch
    x_batch = np.random.rand(10, 8)  # Ten volumes with 8 slices each

    # Assert a single float value is yielded
    loss_dist = loss_distance_func(x_batch)
    assert isinstance(loss_dist[0], np.float32)

    # Assert same value are reached with numpy
    x_diff = x_batch[:, 1:] - x_batch[:, 0:-1]
    loss_dist_np = np.sum(np_smooth_l1(x_diff[:, 1:], x_diff[:, 0:-1]))
    np.testing.assert_almost_equal(loss_dist, loss_dist_np, decimal=3)


def test_distance_loss_equidistant(loss_distance_func):

    # Assert equally spaced scores give proper values
    a = np.array(range(1, 9))
    x_batch = np.tile(a, [10, 1])
    x_diff = x_batch[:, 1:] - x_batch[:, 0:-1]
    loss_dist = loss_distance_func([x_batch])
    loss_dist = loss_dist[0]

    loss_dist_np = np.sum(np_smooth_l1(x_diff[:, 1:], x_diff[:, 0:-1]))
    np.testing.assert_almost_equal(loss_dist, loss_dist_np, decimal=3)
