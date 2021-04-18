import os
from typing import Callable, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/'
CHANNEL_AXIS = 1 if backend.image_data_format() == 'channels_first' else -1


def mobilenet_v3_small(include_top: bool = True, weights: str = 'imagenet', alpha: float = 1.0,
                       minimalistic: bool = False, input_shape: tuple = (224, 224, 3), classes: int = 1000) -> Model:
    if minimalistic:
        kernel, activation, se_ratio = 3, relu, None
    else:
        kernel, activation, se_ratio = 5, hard_swish, 0.25

    def depth(d):
        return make_divisible(d * alpha)

    inputs = layers.Input(input_shape)

    x = stem(inputs, activation)

    x = inverted_res_block(x, depth(16), expansion=1, kernel_size=3, strides=2, se_ratio=se_ratio,
                           activation=relu, name='expanded_conv')
    x = inverted_res_block(x, depth(24), expansion=72.0 / 16, kernel_size=3, strides=2, se_ratio=None,
                           activation=relu, name='expanded_conv_1')
    x = inverted_res_block(x, depth(24), expansion=88.0 / 24, kernel_size=3, strides=1, se_ratio=None,
                           activation=relu, name='expanded_conv_2')
    x = inverted_res_block(x, depth(40), expansion=4, kernel_size=kernel, strides=2, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_3')
    x = inverted_res_block(x, depth(40), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_4')
    x = inverted_res_block(x, depth(40), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_5')
    x = inverted_res_block(x, depth(48), expansion=3, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_6')
    x = inverted_res_block(x, depth(48), expansion=3, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_7')
    x = inverted_res_block(x, depth(96), expansion=6, kernel_size=kernel, strides=2, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_8')
    x = inverted_res_block(x, depth(96), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_9')
    x = inverted_res_block(x, depth(96), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_10')

    x = head(x, 1024, activation=activation, include_top=include_top, classes=classes, dropout_rate=0.2)

    model = Model(inputs, x,  name='MobilenetV3small')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights('small', alpha, minimalistic, include_top)
        model.load_weights(weight_path)

    return model


def mobilenet_v3_large(include_top: bool = True, weights: str = 'imagenet', alpha: float = 1.0,
                       minimalistic: bool = False, input_shape: tuple = (224, 224, 3), classes: int = 1000) -> Model:
    if minimalistic:
        kernel, activation, se_ratio = 3, relu, None
    else:
        kernel, activation, se_ratio = 5, hard_swish, 0.25

    def depth(d):
        return make_divisible(d * alpha)

    inputs = layers.Input(input_shape)

    x = stem(inputs, activation)

    x = inverted_res_block(x, depth(16), expansion=1, kernel_size=3, strides=1, se_ratio=None,
                           activation=relu, name='expanded_conv')
    x = inverted_res_block(x, depth(24), expansion=4, kernel_size=3, strides=2, se_ratio=None,
                           activation=relu, name='expanded_conv_1')
    x = inverted_res_block(x, depth(24), expansion=3, kernel_size=3, strides=1, se_ratio=None,
                           activation=relu, name='expanded_conv_2')
    x = inverted_res_block(x, depth(40), expansion=3, kernel_size=kernel, strides=2, se_ratio=se_ratio,
                           activation=relu, name='expanded_conv_3')
    x = inverted_res_block(x, depth(40), expansion=3, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=relu, name='expanded_conv_4')
    x = inverted_res_block(x, depth(40), expansion=3, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=relu, name='expanded_conv_5')
    x = inverted_res_block(x, depth(80), expansion=6, kernel_size=3, strides=2, se_ratio=None,
                           activation=activation, name='expanded_conv_6')
    x = inverted_res_block(x, depth(80), expansion=2.5, kernel_size=3, strides=1, se_ratio=None,
                           activation=activation, name='expanded_conv_7')
    x = inverted_res_block(x, depth(80), expansion=2.3, kernel_size=3, strides=1, se_ratio=None,
                           activation=activation, name='expanded_conv_8')
    x = inverted_res_block(x, depth(80), expansion=2.3, kernel_size=3, strides=1, se_ratio=None,
                           activation=activation, name='expanded_conv_9')
    x = inverted_res_block(x, depth(112), expansion=6, kernel_size=3, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_10')
    x = inverted_res_block(x, depth(112), expansion=6, kernel_size=3, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_11')
    x = inverted_res_block(x, depth(160), expansion=6, kernel_size=kernel, strides=2, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_12')
    x = inverted_res_block(x, depth(160), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_13')
    x = inverted_res_block(x, depth(160), expansion=6, kernel_size=kernel, strides=1, se_ratio=se_ratio,
                           activation=activation, name='expanded_conv_14')
    x = head(x, 1280, activation=activation, include_top=include_top, classes=classes, dropout_rate=0.2)

    model = Model(inputs, x,  name='MobilenetV3large')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights('large', alpha, minimalistic, include_top)
        model.load_weights(weight_path)

    return model


def stem(inputs: tf.Tensor, activation) -> tf.Tensor:
    # x = layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1.0)(inputs)
    x = layers.experimental.preprocessing.Rescaling(scale=1.0 / 255.)(inputs)

    x = layers.Conv2D(16, 3, strides=2, padding='same', use_bias=False, name='Conv')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name=f'Conv/BatchNorm')(x)
    x = activation(x)

    return x


def head(inputs: tf.Tensor, filters: int, activation: Callable, include_top: bool, classes: int,
         dropout_rate: Union[float, None]) -> tf.Tensor:
    last_conv_filters = make_divisible(backend.int_shape(inputs)[CHANNEL_AXIS] * 6)

    x = layers.Conv2D(last_conv_filters, 1, padding='same', use_bias=False, name='Conv_1')(inputs)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999,
                                  name='Conv_1/BatchNorm')(x)
    x = activation(x)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=True, name='Conv_2')(x)
    x = activation(x)

    x = layers.GlobalAveragePooling2D()(x)

    if include_top:
        pool_shape = (filters, 1, 1) if CHANNEL_AXIS == 1 else (1, 1, filters)
        x = layers.Reshape(pool_shape)(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)

        x = layers.Conv2D(classes, kernel_size=1, padding='same', name='Logits')(x)
        x = layers.Flatten()(x)
        x = layers.Activation(activation='softmax', name='Predictions')(x)

    return x


def se_block(inputs: tf.Tensor, filters: int, se_ratio: float, name: str) -> tf.Tensor:
    x = layers.GlobalAveragePooling2D(name=f'{name}/squeeze_excite/AvgPool')(inputs)

    se_shape = (filters, 1, 1) if CHANNEL_AXIS == 1 else (1, 1, filters)
    x = layers.Reshape(se_shape)(x)

    se_filters = make_divisible(filters * se_ratio)
    x = layers.Conv2D(se_filters, 1, padding='same', name=f'{name}/squeeze_excite/Conv')(x)
    x = layers.ReLU(name=f'{name}/squeeze_excite/Relu')(x)

    x = layers.Conv2D(filters, 1, padding='same', name=f'{name}/squeeze_excite/Conv_1')(x)
    x = hard_sigmoid(x)

    x = layers.Multiply(name=f'{name}/squeeze_excite/Mul')([inputs, x])

    return x


def inverted_res_block(inputs: tf.Tensor, filters: int, expansion: float, kernel_size: int, strides: int,
                       se_ratio: Union[float, None], activation, name: str) -> tf.Tensor:
    input_filters = backend.int_shape(inputs)[CHANNEL_AXIS]
    x = inputs
    if expansion != 1:
        expand_filters = make_divisible(input_filters * expansion)
        x = layers.Conv2D(expand_filters, 1, padding='same', use_bias=False, name=f'{name}/expand')(x)
        x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name=f'{name}/expand/BatchNorm')(
            x)
        x = activation(x)
    else:
        expand_filters = input_filters

    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False,
                               name=f'{name}/depthwise')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999,
                                  name=f'{name}/depthwise/BatchNorm')(x)
    x = activation(x)

    if se_ratio:
        x = se_block(x, expand_filters, se_ratio, name)

    x = layers.Conv2D(filters, 1, padding='same', use_bias=False, name=f'{name}/project')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999,
                                  name=f'{name}/project/BatchNorm')(x)

    if strides == 1 and input_filters == filters:
        x = layers.Add(name=f'{name}/Add')([inputs, x])

    return x


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def download_imagenet_weights(model_type: str, alpha: float, minimalistic: bool, include_top: bool):
    model_name = f'weights_mobilenet_v3_{model_type}{"_minimalistic" if minimalistic else ""}_224_{alpha}_float'
    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}_no_top.h5'

    weight_path = tf.keras.utils.get_file(
        os.path.basename(weight_url),
        weight_url,
        cache_dir='~/.keras',
        cache_subdir='models',
    )
    return weight_path


def preprocess(image_data):
    image_data = image_data.astype(np.float32)
    return image_data
