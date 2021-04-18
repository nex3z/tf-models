import os
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/inception_resnet_v2/'
CHANNEL_AXIS = 1 if backend.image_data_format() == 'channels_first' else -1


def inception_resnet_v2(
        include_top: bool = True,
        weights: str = 'imagenet',
        input_shape: tuple = (299, 299, 3),
        classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    x = conv2d_bn(inputs, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling2D(3, strides=2)(x)

    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    x = layers.Concatenate(axis=CHANNEL_AXIS, name='mixed_5b')([branch_0, branch_1, branch_2, branch_pool])

    for block_idx in range(1, 11):
        x = inception_resnet_a(x, scale=0.17, name=f'block35_{block_idx}')

    x = reduction_a(x, name='mixed_6a')

    for block_idx in range(1, 21):
        x = inception_resnet_b(x, scale=0.1, name=f'block17_{block_idx}')

    x = reduction_b(x, name='mixed_7a')

    for block_idx in range(1, 10):
        x = inception_resnet_c(x, scale=0.2, name=f'block8_{block_idx}')
    x = inception_resnet_c(x, scale=1.0, activation=None, name=f'block8_10')

    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x)

    if weights == 'imagenet':
        weight_path = download_imagenet_weights(include_top)
        model.load_weights(weight_path)

    return model


def inception_resnet_a(inputs: tf.Tensor, scale: float, activation: str = 'relu', name: str = None) -> tf.Tensor:
    branch_0 = conv2d_bn(inputs, 32, 1)

    branch_1 = conv2d_bn(inputs, 32, 1)
    branch_1 = conv2d_bn(branch_1, 32, 3)

    branch_2 = conv2d_bn(inputs, 32, 1)
    branch_2 = conv2d_bn(branch_2, 48, 3)
    branch_2 = conv2d_bn(branch_2, 64, 3)

    x = residual_scaling(inputs, [branch_0, branch_1, branch_2], scale, name)

    if activation is not None:
        x = layers.Activation(activation, name=f'{name}_ac')(x)

    return x


def reduction_a(inputs: tf.Tensor, name: str) -> tf.Tensor:
    branch_0 = conv2d_bn(inputs, 384, 3, strides=2, padding='valid')

    branch_1 = conv2d_bn(inputs, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')

    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(inputs)

    x = layers.Concatenate(axis=CHANNEL_AXIS, name=name)([branch_0, branch_1, branch_pool])
    return x


def inception_resnet_b(inputs: tf.Tensor, scale: float, activation: str = 'relu', name: str = None) -> tf.Tensor:
    branch_0 = conv2d_bn(inputs, 192, 1)

    branch_1 = conv2d_bn(inputs, 128, 1)
    branch_1 = conv2d_bn(branch_1, 160, (1, 7))
    branch_1 = conv2d_bn(branch_1, 192, (7, 1))

    x = residual_scaling(inputs, [branch_0, branch_1], scale, name)

    if activation is not None:
        x = layers.Activation(activation, name=f'{name}_ac')(x)

    return x


def reduction_b(inputs: tf.Tensor, name: str) -> tf.Tensor:
    branch_0 = conv2d_bn(inputs, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')

    branch_1 = conv2d_bn(inputs, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')

    branch_2 = conv2d_bn(inputs, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')

    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(inputs)

    x = layers.Concatenate(axis=CHANNEL_AXIS, name=name)([branch_0, branch_1, branch_2, branch_pool])
    return x


def inception_resnet_c(inputs: tf.Tensor, scale: float, activation: Union[str, None] = 'relu',
                       name: str = None) -> tf.Tensor:
    branch_0 = conv2d_bn(inputs, 192, 1)

    branch_1 = conv2d_bn(inputs, 192, 1)
    branch_1 = conv2d_bn(branch_1, 224, (1, 3))
    branch_1 = conv2d_bn(branch_1, 256, (3, 1))

    x = residual_scaling(inputs, [branch_0, branch_1], scale, name)

    if activation is not None:
        x = layers.Activation(activation, name=f'{name}_ac')(x)

    return x


def conv2d_bn(inputs: tf.Tensor, filters: int, kernel_size: Union[int, Tuple], strides: int = 1, padding: str = 'same',
              name: str = None) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(inputs)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, scale=False, name=f'{name}_bn' if name is not None else None)(x)
    x = layers.Activation('relu', name=f'{name}_ac' if name is not None else None)(x)
    return x


def residual_scaling(inputs: tf.Tensor, branches: List[tf.Tensor], scale: float, name: str = None):
    mixed = layers.Concatenate(axis=CHANNEL_AXIS, name=f'{name}_mixed')(branches)

    input_shape = backend.int_shape(inputs)
    up = layers.Conv2D(input_shape[CHANNEL_AXIS], 1, activation=None, use_bias=True, name=f'{name}_conv')(mixed)

    x = layers.Lambda(
        lambda _inputs, _scale: _inputs[0] + _inputs[1] * _scale,
        output_shape=input_shape[1:],
        arguments={'_scale': scale},
        name=name
    )([inputs, up])

    return x


def download_imagenet_weights(include_top: bool):
    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

    weight_path = tf.keras.utils.get_file(
        os.path.basename(weight_url),
        weight_url,
        cache_dir='~/.keras',
        cache_subdir='models',
    )
    return weight_path


def preprocess(image_data):
    image_data = image_data.astype(np.float32)
    image_data /= 127.5
    image_data -= 1.0
    return image_data
