import dataclasses
import math
import os
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

from config import BlockConfig, DEFAULT_BLOCKS_ARGS

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/keras-applications/'
CHANNEL_AXIS = 1 if backend.image_data_format() == 'channels_first' else 3

CONV_K_INIT = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_K_INIT = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def efficientnet_b0(include_top: bool = True, weights: str = 'imagenet',
                    input_shape: tuple = (224, 224, 3), classes: int = 1000):
    return efficientnet(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        classes=classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        configs=DEFAULT_BLOCKS_ARGS,
        dropout_rate=0.2,
        model_name='efficientnetb0',
    )


def efficientnet_b5(include_top: bool = True, weights: str = 'imagenet',
                    input_shape: tuple = (456, 456, 3), classes: int = 1000):
    return efficientnet(
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        classes=classes,
        width_coefficient=1.6,
        depth_coefficient=2.2,
        configs=DEFAULT_BLOCKS_ARGS,
        dropout_rate=0.4,
        model_name='efficientnetb5',
    )


def efficientnet(
        include_top: bool = True,
        weights: str = 'imagenet',
        input_shape: tuple = (224, 224, 3),
        classes: int = 1000,
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        configs: List[BlockConfig] = DEFAULT_BLOCKS_ARGS,
        drop_connect_rate: float = 0.2,
        dropout_rate: float = 0.2,
        activation: str = 'swish',
        model_name: str = 'efficientnet'):
    inputs = layers.Input(shape=input_shape)
    # Preprocess
    x = layers.experimental.preprocessing.Rescaling(1. / 255.)(inputs)
    x = layers.experimental.preprocessing.Normalization(axis=CHANNEL_AXIS)(x)

    # Stem
    x = layers.Conv2D(round_filters(32, width_coefficient), 3, strides=2, padding='same', use_bias=False,
                      kernel_initializer=CONV_K_INIT, name='stem_conv')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, name='stem_bn')(x)
    x = layers.Activation(activation, name='stem_activation')(x)

    # Blocks
    num_blocks = float(sum(round_repeats(config.repeats, depth_coefficient) for config in configs))
    b = 0
    for i, config in enumerate(configs):
        num_filters_in = round_filters(config.filters_in, width_coefficient)
        num_filters_out = round_filters(config.filters_out, width_coefficient)
        for j in range(round_repeats(config.repeats, depth_coefficient)):
            block_config = dataclasses.replace(
                config,
                strides=config.strides if j == 0 else 1,
                filters_in=num_filters_in if j == 0 else num_filters_out,
                filters_out=num_filters_out,
                drop_rate=drop_connect_rate * b / num_blocks,
            )
            x = mbconv_block(x, block_config, name=f'block_{i + 1}{chr(j + 97)}')

    # Top
    x = layers.Conv2D(round_filters(1280, width_coefficient), 1, padding='same', use_bias=False,
                      kernel_initializer=CONV_K_INIT, name='top_conv')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, name='top_bn')(x)
    x = layers.Activation(activation, name='top_activation')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name='top_dropout')(x)
        x = layers.Dense(classes, activation='softmax', kernel_initializer=DENSE_K_INIT, name='predictions')(x)

    model = Model(inputs, x, name=model_name)

    if weights == 'imagenet':
        weight_path = download_imagenet_weights(model_name, include_top)
        model.load_weights(weight_path)

    return model


def mbconv_block(inputs: tf.Tensor, config: BlockConfig, name: str) -> tf.Tensor:
    x = inputs
    num_filters_expanded = config.filters_in * config.expand_ratio
    if config.expand_ratio != 1:
        # Expand
        x = layers.Conv2D(num_filters_expanded, 1, padding='same', use_bias=False,
                          kernel_initializer=CONV_K_INIT, name=f'{name}_expand_conv')(x)
        x = layers.BatchNormalization(axis=CHANNEL_AXIS, name=f'{name}_expand_bn')(x)
        x = layers.Activation(config.activation, name=f'{name}_expand_activation')(x)

    # Depthwise
    x = layers.DepthwiseConv2D(config.kernel_size, strides=config.strides, padding='same', use_bias=False,
                               depthwise_initializer=CONV_K_INIT, name=f'{name}_dwconv')(x)
    x = layers.BatchNormalization(CHANNEL_AXIS, name=f'{name}_bn')(x)
    x = layers.Activation(config.activation, name=f'{name}_activation')(x)

    # Squeeze and Excitation
    if 0 < config.se_ratio <= 1:
        filters_se = max(1, int(config.filters_in * config.se_ratio))
        se = layers.GlobalAveragePooling2D(name=f'{name}_se_squeeze')(x)
        se_shape = (num_filters_expanded, 1, 1) if CHANNEL_AXIS == 1 else (1, 1, num_filters_expanded)
        se = layers.Reshape(se_shape, name=f'{name}_se_reshape')(se)
        se = layers.Conv2D(filters_se, 1, padding='same', activation=config.activation,
                           kernel_initializer=CONV_K_INIT, name=f'{name}_se_reduce')(se)
        se = layers.Conv2D(num_filters_expanded, 1, padding='same', activation='sigmoid',
                           kernel_initializer=CONV_K_INIT, name=f'{name}_se_expand')(se)
        x = layers.multiply([x, se], name=f'{name}_se_excite')

    # Output
    x = layers.Conv2D(config.filters_out, 1, padding='same', use_bias=False,
                      kernel_initializer=CONV_K_INIT, name=f'{name}_project_conv')(x)
    x = layers.BatchNormalization(axis=CHANNEL_AXIS, name=f'{name}_project_bn')(x)

    # Skip connection
    if config.id_skip and config.strides == 1 and config.filters_in == config.filters_out:
        if config.drop_rate > 0:
            x = layers.Dropout(config.drop_rate, noise_shape=(None, 1, 1, 1), name=f'{name}_drop')(x)
        x = layers.add([x, inputs], name=f'{name}_add')

    return x


def round_filters(filters: int, width_coefficient: float, divisor: int = 8):
    """Round number of filters based on depth multiplier."""
    filters *= width_coefficient
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats: int, depth_coefficient: float):
    """Round number of repeats based on depth multiplier."""
    return int(math.ceil(depth_coefficient * repeats))


def download_imagenet_weights(model_name: str, include_top: bool):
    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}_no_top.h5'

    weight_path = tf.keras.utils.get_file(
        os.path.basename(weight_url),
        weight_url,
        cache_dir='./weights',
        cache_subdir='',
    )
    return weight_path


def preprocess(image_data):
    image_data = image_data.astype(np.float32)
    return image_data
