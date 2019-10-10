from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, TimeDistributed, GlobalMaxPooling2D
from keras.optimizers import rmsprop
from ssbr.loss import loss_ssbr, loss_distance, loss_order


def vgg16_features():
    # load vgg16 and take first X convs
    vgg = VGG16(include_top=False, weights='imagenet', pooling=None)
    return vgg


def ssbr_model(input_shape=(None, None, 3), num_slices=8, lr=0.0001, batch_size=None, alpha=0.5):
    # Define stack input
    inp_stack = Input(batch_shape=(
        batch_size,
        num_slices,
    ) + input_shape, name='input_stack')

    # Define score extractor model
    inp = Input(shape=input_shape, name='input_slice')
    vgg = vgg16_features()
    x = vgg(inp)

    x = Conv2D(512, (1, 1), activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    # x = GlobalMaxPooling2D()(x)

    single_score = Dense(1, name='output_slice')(x)

    # Wrap score extractor in TimeDistributed
    score_extractor = Model(inp, single_score)
    scores = TimeDistributed(score_extractor, input_shape=(num_slices, *input_shape), name='output_stack')(inp_stack)

    # Define losses
    m = Model(inp_stack, scores)
    opt = rmsprop(lr=lr)

    l = loss_ssbr(alpha=alpha)
    m.compile(loss=l, optimizer=opt, metrics=[loss_order, loss_distance])

    return m, score_extractor