# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains common building blocks for neural networks."""

from typing import Sequence, Union

# Import libraries
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.beta.modeling.layers import nn_layers


@tf.keras.utils.register_keras_serializable(package='Vision')
class BasicBlock2DVolume(tf.keras.layers.Layer):
  """A basic 2d convolution block."""

  def __init__(self,
               filters: Union[int, Sequence[int]],
               strides: Union[int, Sequence[int]],
               kernel_size: Union[int, Sequence[int]],
               kernel_initializer: str = 'VarianceScaling',
               kernel_regularizer: tf.keras.regularizers.Regularizer = None,
               bias_regularizer: tf.keras.regularizers.Regularizer = None,
               activation: str = 'relu',
               use_sync_bn: bool = False,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               use_batch_normalization: bool = False,
               **kwargs):
    """Creates a basic 2d convolution block applying one or more convolutions.

    Args:
      filters: A list of `int` numbers or an `int` number of filters. Given an
        `int` input, a single convolution is applied; otherwise a series of
        convolutions are applied.
      strides: An integer or tuple/list of 3 integers, specifying the strides of
        the convolution along each spatial dimension. Can be a single integer to
        specify the same value for all spatial dimensions.
      kernel_size: An integer or tuple/list of 3 integers, specifying the depth,
        height and width of the 3D convolution window. Can be a single integer
        to specify the same value for all spatial dimensions.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      use_batch_normalization: Wheher to use batch normalizaion or not.
      **kwargs: keyword arguments to be passed.
    """

    super().__init__(**kwargs)

    if isinstance(filters, int):
      self._filters = [filters]
    else:
      self._filters = filters
    self._strides = strides
    self._kernel_size = kernel_size
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._use_batch_normalization = use_batch_normalization

    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._activation_fn = tf_utils.get_activation(activation)

  def build(self, input_shape: tf.TensorShape):
    """Builds the basic 2d convolution block."""
    self._convs = []
    self._norms = []
    for filters in self._filters:
      self._convs.append(
          tf.keras.layers.Conv2D(
              filters=filters,
              kernel_size=self._kernel_size,
              strides=self._strides,
              padding='same',
              data_format=tf.keras.backend.image_data_format(),
              activation=None))
      self._norms.append(
          self._norm(
              axis=self._bn_axis,
              momentum=self._norm_momentum,
              epsilon=self._norm_epsilon))

    super(BasicBlock2DVolume, self).build(input_shape)

  def get_config(self):
    """Returns the config of the basic 2d convolution block."""
    config = {
        'filters': self._filters,
        'strides': self._strides,
        'kernel_size': self._kernel_size,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'use_batch_normalization': self._use_batch_normalization
    }
    base_config = super(BasicBlock2DVolume, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
    """Runs forward pass on the input tensor."""
    x = inputs
    for conv, norm in zip(self._convs, self._norms):
      x = conv(x)
      if self._use_batch_normalization:
        x = norm(x)
      x = self._activation_fn(x)
    return x
