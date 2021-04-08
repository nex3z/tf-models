import logging
import os
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/'
CHANNEL_AXIS = 1 if backend.image_data_format() == 'channels_first' else -1


def mobilenet_v2(include_top: bool = True, weights: str = 'imagenet', alpha: float = 1.0,
                 input_shape: tuple = (224, 224, 3), classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    first_block_filters = make_divisible(32 * alpha, 8)
    x = layers.Conv2D(first_block_filters, 3, strides=2, padding='same', use_bias=False, name='Conv1')(inputs)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
    x = layers.ReLU(6.0, name='Conv1_relu')(x)

    x = stack(x, expansion=1, filters=16, blocks=1, alpha=alpha, strides1=1, names=['expanded_conv'])
    x = stack(x, expansion=6, filters=24, blocks=2, alpha=alpha, strides1=2, names=build_block_names(1, 2))
    x = stack(x, expansion=6, filters=32, blocks=3, alpha=alpha, strides1=2, names=build_block_names(3, 5))
    x = stack(x, expansion=6, filters=64, blocks=4, alpha=alpha, strides1=2, names=build_block_names(6, 9))
    x = stack(x, expansion=6, filters=96, blocks=3, alpha=alpha, strides1=1, names=build_block_names(10, 12))
    x = stack(x, expansion=6, filters=160, blocks=3, alpha=alpha, strides1=2, names=build_block_names(13, 15))
    x = stack(x, expansion=6, filters=320, blocks=1, alpha=alpha, strides1=1, names=build_block_names(16))

    last_block_filters = make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280
    x = layers.Conv2D(last_block_filters, 1, use_bias=False, name='Conv_1')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
    x = layers.ReLU(6.0, name='out_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    rows = input_shape[0]
    model = Model(inputs, x, name=f'mobilenetv2_{alpha:.2f}_{rows}')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights(alpha, input_shape, include_top)
        model.load_weights(weight_path)

    return model


def build_block_names(start, end=None):
    if end is None:
        end = start
    return [f'block_{i}' for i in range(start, end + 1)]


def stack(inputs: tf.Tensor, expansion: int, filters: int, blocks: int, strides1: int, alpha: float,
          names: List[str]) -> tf.Tensor:
    x = inverted_res_block(inputs, expansion=expansion, filters=filters, strides=strides1, alpha=alpha, name=names[0])
    for i in range(1, blocks):
        x = inverted_res_block(x, expansion=expansion, filters=filters, strides=1, alpha=alpha, name=names[i])
    return x


def inverted_res_block(inputs: tf.Tensor, expansion: int, filters: int, strides: int, alpha: float,
                       name: str) -> tf.Tensor:
    x = inputs
    in_channels = backend.int_shape(x)[CHANNEL_AXIS]

    if expansion != 1:
        # Expand
        x = layers.Conv2D(expansion * in_channels, 1, padding='same', use_bias=False, name=f'{name}_expand')(x)
        x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name=f'{name}_expand_BN')(x)
        x = layers.ReLU(6.0, name=f'{name}_expand_relu')(x)

    # Depthwise
    x = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False, name=f'{name}_depthwise')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name=f'{name}_depthwise_BN')(x)
    x = layers.ReLU(6.0, name=f'{name}_depthwise_relu')(x)

    # Project
    pointwise_filters = make_divisible(int(filters * alpha), 8)
    x = layers.Conv2D(pointwise_filters, 1, padding='same', use_bias=False, name=f'{name}_project')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, epsilon=1e-3, momentum=0.999, name=f'{name}_project_BN')(x)
    # Linear bottleneck, no non-linearity here

    if in_channels == pointwise_filters and strides == 1:
        x = layers.Add(name=f'{name}_add')([inputs, x])

    return x


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def download_imagenet_weights(alpha: float, input_shape: tuple, include_top: bool):
    if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
        raise ValueError(f"Invalid alpha = {alpha}, alpha can only be one of [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]")
    rows, cols = input_shape[0], input_shape[1]
    if rows != cols or rows not in [96, 128, 160, 192, 224]:
        logging.warning(f"Invalid input_shape = {input_shape}, "
                        f"input_shape should be square and edge length should be in [96, 128, 160, 192, 224]. "
                        f"Weights for input shape (224, 224) will be loaded as the default.")
        rows = 224

    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{alpha}_{rows}.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}mobilenet_v2_weights_tf_dim_ordering_tf_kernels_{alpha}_{rows}_no_top.h5'

    weight_path = tf.keras.utils.get_file(
        os.path.basename(weight_url),
        weight_url,
        cache_dir='./weights',
        cache_subdir='',
    )
    return weight_path


def preprocess(image_data):
    image_data = image_data.astype(np.float32)
    image_data /= 127.5
    image_data -= 1.0
    return image_data
