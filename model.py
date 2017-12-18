from keras.layers import Activation, Reshape, Dropout
from keras.layers import AtrousConvolution2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.models import Sequential


#
# The VGG16 keras model is taken from here:
# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# The (caffe) structure of DilatedNet is here:
# https://github.com/fyu/dilation/blob/master/models/dilation8_pascal_voc_deploy.prototxt

def get_frontend(input_width, input_height) -> Sequential:
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1', input_shape=(input_width, input_height, 3)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add(AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, 1, 1, activation='relu', name='fc7'))
    model.add(Dropout(0.5))

    # Note: this layer has linear activations, not ReLU
    # model.add(Convolution2D(21, 1, 1, activation='linear', name='fc-final'))

    # Changing the number of channels for Stanford Dataset
    model.add(Convolution2D(9, 1, 1, activation='linear', name='fc-final_2',init='glorot_normal'))

    # model.layers[-1].output_shape == (None, 16, 16, 21)
    return model


def add_softmax(model: Sequential) -> Sequential:
    """ Append the softmax layers to the frontend or frontend + context net. """
    # The softmax layer doesn't work on the (width, height, channel)
    # shape, so we reshape to (width*height, channel) first.
    # https://github.com/fchollet/keras/issues/1169
    _, curr_width, curr_height, curr_channels = model.layers[-1].output_shape

    model.add(Reshape((curr_width * curr_height, curr_channels)))
    model.add(Activation('softmax'))
    # Technically, we need another Reshape here to reshape to 2d, but TF
    # the complains when batch_size > 1. We're just going to reshape in numpy.
    # model.add(Reshape((curr_width, curr_height, curr_channels)))

    return model


def add_context(model: Sequential) -> Sequential:
    """ Append the context layers to the frontend. """
    model.add(ZeroPadding2D(padding=(33, 33)))
    model.add(Convolution2D(18, 3, 3, activation='relu', name='ctx_conv1_1'))
    model.add(Convolution2D(18, 3, 3, activation='relu', name='ctx_conv1_2'))
    model.add(AtrousConvolution2D(36, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1'))
    model.add(AtrousConvolution2D(72, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1'))
    model.add(AtrousConvolution2D(144, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1'))
    model.add(AtrousConvolution2D(288, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1'))
    model.add(Convolution2D(288, 3, 3, activation='relu', name='ctx_fc1'))
    model.add(Convolution2D(9, 1, 1, name='ctx_final'))

    return model
