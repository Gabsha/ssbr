import numpy as np

from ssbr.model import ssbr_model


def test_model():
    m, _ = ssbr_model(input_shape=(64, 64, 3))

    # Define test stack of images
    batch = np.random.rand(2, 8, 64, 64, 3)  # Batch of 2 with 8 RGB slices of 128x128 pixels
    scores = m.predict_on_batch(batch)
    assert scores.shape == (2, 8, 1)


def test_fit():
    m, _ = ssbr_model(input_shape=(64, 64, 3))
    X = np.random.rand(6, 8, 64, 64, 3)  # Batch of 2 with 8 RGB slices of 128x128 pixels
    Y = np.zeros((6, 8, 1))  # Dummy data to fit Keras loss function definition
    m.fit(X, Y, batch_size=2, epochs=1)