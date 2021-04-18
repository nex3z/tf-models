import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/'
CHANNEL_AXIS = 1 if backend.image_data_format() == 'channels_first' else -1


def inception_v3(include_top: bool = True, weights: str = 'imagenet',
                 input_shape: tuple = (299, 299, 3), classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    # Stem
    x = conv2d_bn(inputs, 32, (3, 3), strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, (3, 3), padding='valid')
    x = conv2d_bn(x, 64, (3, 3), padding='same')
    x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, (1, 1), padding='valid')
    x = conv2d_bn(x, 192, (3, 3), padding='valid')
    x = layers.MaxPool2D((3, 3), strides=(2, 2))(x)

    # Blocks
    x = inception_a_block(x, ave_pool_filters=32, name='mixed0')
    x = inception_a_block(x, ave_pool_filters=64, name='mixed1')
    x = inception_a_block(x, ave_pool_filters=64, name='mixed2')

    x = reduction_a_block(x, name='mixed3')

    x = inception_b_block(x, branch_7x7_initial_filters=128, name='mixed4')
    x = inception_b_block(x, branch_7x7_initial_filters=160, name='mixed5')
    x = inception_b_block(x, branch_7x7_initial_filters=160, name='mixed6')
    x = inception_b_block(x, branch_7x7_initial_filters=192, name='mixed7')

    x = reduction_b_block(x)

    x = inception_c_block(x, name='mixed9')
    x = inception_c_block(x, name='mixed10')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='inception_v3')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights(include_top)
        model.load_weights(weight_path)

    return model


def inception_a_block(inputs: tf.Tensor, ave_pool_filters: int = 32, name: str = '') -> tf.Tensor:
    branch_1x1 = conv2d_bn(inputs, 64, (1, 1))

    branch_5x5 = conv2d_bn(inputs, 48, (1, 1))
    branch_5x5 = conv2d_bn(branch_5x5, 64, (5, 5))

    branch_3x3_db1 = conv2d_bn(inputs, 64, (1, 1))
    branch_3x3_db1 = conv2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = conv2d_bn(branch_3x3_db1, 96, (3, 3))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv2d_bn(branch_pool, ave_pool_filters, (1, 1))

    x = layers.concatenate([branch_1x1, branch_5x5, branch_3x3_db1, branch_pool], axis=CHANNEL_AXIS, name=name)
    return x


def reduction_a_block(inputs: tf.Tensor, name: str = '') -> tf.Tensor:
    branch_3x3 = conv2d_bn(inputs, 384, (3, 3), strides=(2, 2), padding='valid')

    branch_3x3_db1 = conv2d_bn(inputs, 64, (1, 1))
    branch_3x3_db1 = conv2d_bn(branch_3x3_db1, 96, (3, 3))
    branch_3x3_db1 = conv2d_bn(branch_3x3_db1, 96, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = layers.concatenate([branch_3x3, branch_3x3_db1, branch_pool], axis=CHANNEL_AXIS, name=name)
    return x


def inception_b_block(inputs: tf.Tensor, branch_7x7_initial_filters: int = 128, name: str = '') -> tf.Tensor:
    branch_1x1 = conv2d_bn(inputs, 192, (1, 1))

    branch_7x7 = conv2d_bn(inputs, branch_7x7_initial_filters, (1, 1))
    branch_7x7 = conv2d_bn(branch_7x7, branch_7x7_initial_filters, (1, 7))
    branch_7x7 = conv2d_bn(branch_7x7, 192, (7, 1))

    branch_7x7_db1 = conv2d_bn(inputs, branch_7x7_initial_filters, (1, 1))
    branch_7x7_db1 = conv2d_bn(branch_7x7_db1, branch_7x7_initial_filters, (7, 1))
    branch_7x7_db1 = conv2d_bn(branch_7x7_db1, branch_7x7_initial_filters, (1, 7))
    branch_7x7_db1 = conv2d_bn(branch_7x7_db1, branch_7x7_initial_filters, (7, 1))
    branch_7x7_db1 = conv2d_bn(branch_7x7_db1, 192, (1, 7))

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

    x = layers.concatenate([branch_1x1, branch_7x7, branch_7x7_db1, branch_pool], axis=CHANNEL_AXIS, name=name)
    return x


def reduction_b_block(inputs: tf.Tensor, name: str = '') -> tf.Tensor:
    branch_3x3 = conv2d_bn(inputs, 192, (1, 1))
    branch_3x3 = conv2d_bn(branch_3x3, 320, (3, 3), strides=(2, 2), padding='valid')

    branch_7x7x3 = conv2d_bn(inputs, 192, (1, 1))
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, (1, 7))
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, (7, 1))
    branch_7x7x3 = conv2d_bn(branch_7x7x3, 192, (3, 3), strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(inputs)

    x = layers.concatenate([branch_3x3, branch_7x7x3, branch_pool], axis=CHANNEL_AXIS, name=name)
    return x


def inception_c_block(inputs: tf.Tensor, name: str = '') -> tf.Tensor:
    branch_1x1 = conv2d_bn(inputs, 320, (1, 1))

    branch_3x3 = conv2d_bn(inputs, 384, (1, 1))
    branch_3x3_1 = conv2d_bn(branch_3x3, 384, (1, 3))
    branch_3x3_2 = conv2d_bn(branch_3x3, 384, (3, 1))
    branch_3x3 = layers.concatenate([branch_3x3_1, branch_3x3_2], axis=CHANNEL_AXIS, name=f'{name}_1')

    branch_3x3_db1 = conv2d_bn(inputs, 448, (1, 1))
    branch_3x3_db1 = conv2d_bn(branch_3x3_db1, 384, (3, 3))
    branch_3x3_db1_1 = conv2d_bn(branch_3x3_db1, 384, (1, 3))
    branch_3x3_db1_2 = conv2d_bn(branch_3x3_db1, 384, (3, 1))
    branch_3x3_db1 = layers.concatenate([branch_3x3_db1_1, branch_3x3_db1_2], axis=CHANNEL_AXIS, name=f'{name}_2')

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
    branch_pool = conv2d_bn(branch_pool, 192, (1, 1))

    x = layers.concatenate([branch_1x1, branch_3x3, branch_3x3_db1, branch_pool], axis=CHANNEL_AXIS, name=name)
    return x


def conv2d_bn(x: tf.Tensor, filters: int, kernel_size: Tuple, strides: Tuple = (1, 1),
              padding: str = 'same') -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, scale=False)(x)
    x = layers.Activation('relu')(x)
    return x


def download_imagenet_weights(include_top: bool):
    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

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
