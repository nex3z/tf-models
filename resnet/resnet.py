import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, backend, Model

BASE_WEIGHTS_URL = 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/'
BN_AXIS = 3 if backend.image_data_format() == 'channels_last' else 1
EPSILON = 1.001e-5


def resnet50(include_top: bool = True, weights: str = 'imagenet', input_shape: tuple = (224, 224, 3),
             classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    x = conv1(inputs)
    x = conv2_pool(x)
    x = stack(x, 64, 3, strides1=1, name='conv2')
    x = stack(x, 128, 4, name='conv3')
    x = stack(x, 256, 6, name='conv4')
    x = stack(x, 512, 3, name='conv5')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='resnet50')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights('resnet50', include_top)
        model.load_weights(weight_path)

    return model


def resnet101(include_top: bool = True, weights: str = 'imagenet', input_shape: tuple = (224, 224, 3),
              classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    x = conv1(inputs)
    x = conv2_pool(x)
    x = stack(x, 64, 3, strides1=1, name='conv2')
    x = stack(x, 128, 4, name='conv3')
    x = stack(x, 256, 23, name='conv4')
    x = stack(x, 512, 3, name='conv5')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='resnet101')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights('resnet101', include_top)
        model.load_weights(weight_path)

    return model


def resnet152(include_top: bool = True, weights: str = 'imagenet', input_shape: tuple = (224, 224, 3),
               classes: int = 1000) -> Model:
    inputs = layers.Input(shape=input_shape)

    x = conv1(inputs)
    x = conv2_pool(x)
    x = stack(x, 64, 3, strides1=1, name='conv2')
    x = stack(x, 128, 8, name='conv3')
    x = stack(x, 256, 36, name='conv4')
    x = stack(x, 512, 3, name='conv5')

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(inputs, x, name='resnet152')

    if weights == 'imagenet':
        weight_path = download_imagenet_weights('resnet152', include_top)
        model.load_weights(weight_path)

    return model


def conv1(x: tf.Tensor):
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(x)
    x = layers.Conv2D(64, 7, strides=2, name='conv1_conv')(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=EPSILON, name='conv1_bn')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    return x


def conv2_pool(x: tf.Tensor):
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)
    return x


def stack(x: tf.Tensor, filters: int, blocks: int, strides1: int = 2, name: str = None) -> tf.Tensor:
    x = block(x, filters, strides=strides1, name=f'{name}_block1')
    for i in range(2, blocks + 1):
        x = block(x, filters, conv_shortcut=False, name=f'{name}_block1{i}')
    return x


def block(x: tf.Tensor, filters: int, kernel_size: int = 3, strides: int = 1, conv_shortcut: bool = True,
          name: str = None) -> tf.Tensor:
    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=strides, name=f'{name}_0_conv')(x)
        shortcut = layers.BatchNormalization(axis=BN_AXIS, epsilon=EPSILON, name=f'{name}_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=strides, name=f'{name}_1_conv')(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=EPSILON, name=f'{name}_1_bn')(x)
    x = layers.Activation('relu', name=f'{name}_1_relu')(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', name=f'{name}_2_conv')(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=EPSILON, name=f'{name}_2_bn')(x)
    x = layers.Activation('relu', name=f'{name}_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, strides=1, name=f'{name}_3_conv')(x)
    x = layers.BatchNormalization(axis=BN_AXIS, epsilon=EPSILON, name=f'{name}_3_bn')(x)
    # No Activation here

    x = layers.Add(name=f'{name}_add')([shortcut, x])
    x = layers.Activation('relu', name=f'{name}_out')(x)

    return x


def download_imagenet_weights(model_name: str, include_top: bool):
    if include_top:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}_weights_tf_dim_ordering_tf_kernels.h5'
    else:
        weight_url = f'{BASE_WEIGHTS_URL}{model_name}_weights_tf_dim_ordering_tf_kernels_notop.h5'

    weight_path = tf.keras.utils.get_file(
        os.path.basename(weight_url),
        weight_url,
        cache_dir='./weights',
        cache_subdir='',
    )
    return weight_path


def preprocess(image_data):
    image_data = image_data.astype(np.float32)
    mean = [103.939, 116.779, 123.68]
    image_data = image_data[..., ::-1]
    image_data[..., 0] -= mean[0]
    image_data[..., 1] -= mean[1]
    image_data[..., 2] -= mean[2]
    return image_data
