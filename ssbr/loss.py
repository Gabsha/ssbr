import keras.backend as K

HUBER_DELTA = 0.5


def smoothL1(y_true, y_pred):

    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x**2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return K.sum(x)


def loss_order(scores_true, scores_pred):
    """
    Implements a ordering loss for slice scores.
    Scores is a tensor of dimension B x S where B is the number of volumes in a batch and S is the number of slice
    scores per volume. The score should have ascending order

    :param scores_pred: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: order loss tensor
    """

    score_diff = scores_pred[:, 1:] - scores_pred[:, 0:-1]
    loss = -K.sum(K.log(K.sigmoid(score_diff) + K.epsilon()))
    return loss


def loss_distance(scores_true, scores_pred):
    """Implements a distance loss between slice scores.
    Scores is a tensor of dimension B x S. Each slices should have equidistant scores, since all input images should be
    equidistant.

    :param scores_true: Not used
    :param scores_pred: Tensor of dimension B x S, where B is the number of volumes and S the number of slice scores.
    :type scores_pred:
    :return: distance loss tensor
    """
    score_diff = scores_pred[:, 1:] - scores_pred[:, 0:-1]
    loss = K.sum(smoothL1(score_diff[:, 1:], score_diff[:, 0:-1]))
    return loss


def loss_ssbr(alpha=0.5):
    def l(scores_true, scores_pred, alpha=0.5):
        return alpha * loss_distance(scores_true, scores_pred) + (1 - alpha) * loss_order(scores_true, scores_pred)

    return l