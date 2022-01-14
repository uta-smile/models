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

"""Data parser and processing for 3D segmentation datasets."""

from typing import Any, Dict, Sequence, Tuple
import tensorflow as tf
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
import numpy as np

from official.projects.volumetric_models.np_data_augmentations.dataset_loading import DataLoader3D
from official.projects.volumetric_models.np_data_augmentations.data_augmentations import tr_transforms, val_transforms


class Decoder(decoder.Decoder):
  """A tf.Example decoder for segmentation task."""

  def __init__(self,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               image_shape_key: str = 'image_shape',
               label_shape_key: str = 'label_shape'
               ):
    self._keys_to_features = {
        image_field_key: tf.io.VarLenFeature(dtype=tf.float32),
        label_field_key: tf.io.VarLenFeature(dtype=tf.int64),
        image_shape_key: tf.io.FixedLenFeature([4], tf.int64),
        label_shape_key: tf.io.FixedLenFeature([3], tf.int64)
    }

  def decode(self, serialized_example: tf.string) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(serialized_example,
                                      self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               input_size: Sequence[int] = [40, 56, 40],
               num_classes: int = 3,
               num_channels: int = 1,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               image_shape_key: str = 'image_shape',
               label_shape_key: str = 'label_shape',
               dtype: str = 'float32',
               label_dtype: str = 'int32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      input_size: The input tensor size of [height, width, volume] of input
        image.
      num_classes: The number of classes to be segmented.
      num_channels: The channel of input images.
      image_field_key: A `str` of the key name to encoded image in TFExample.
      label_field_key: A `str` of the key name to label in TFExample.
      dtype: The data type. One of {`bfloat16`, `float32`, `float16`}.
      label_dtype: The data type of input label.
    """
    self._input_size = input_size
    self._num_classes = num_classes
    self._num_channels = num_channels
    self._image_field_key = image_field_key
    self._label_field_key = label_field_key
    self._image_shape_key = image_shape_key
    self._label_shape_key = label_shape_key
    self._dtype = dtype
    self._label_dtype = label_dtype

  def _prepare_image_and_label_tr(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = data[self._image_field_key]
    if isinstance(image, tf.SparseTensor):
      image = tf.sparse.to_dense(image)

    label = data[self._label_field_key]
    if isinstance(label, tf.SparseTensor):
      label = tf.sparse.to_dense(label)

    image_size = data[self._image_shape_key]
    image = tf.reshape(image, image_size)

    label_size = data[self._label_shape_key]
    label = tf.reshape(label, label_size)
    label = tf.cast(label, dtype=tf.float32)

    image, label = tf.py_function(func=self._data_augmentation_tr, inp=[image, label], Tout=(tf.float32, tf.float32))

    image.set_shape(self._input_size+[self._num_channels])
    label.set_shape(self._input_size+[self._num_classes])

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._label_dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _prepare_image_and_label_val(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = data[self._image_field_key]
    if isinstance(image, tf.SparseTensor):
      image = tf.sparse.to_dense(image)

    label = data[self._label_field_key]
    if isinstance(label, tf.SparseTensor):
      label = tf.sparse.to_dense(label)

    image_size = data[self._image_shape_key]
    image = tf.reshape(image, image_size)

    label_size = data[self._label_shape_key]
    label = tf.reshape(label, label_size)
    label = tf.cast(label, dtype=tf.float32)

    image, label = tf.py_function(func=self._data_augmentation_val, inp=[image, label], Tout=(tf.float32, tf.float32))

    image.set_shape(self._input_size+[self._num_channels])
    label.set_shape(self._input_size+[self._num_classes])

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._label_dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _data_augmentation_tr(self, image, label):
    image = image.numpy()
    label = label.numpy()
    label = label[np.newaxis, :]
    data = np.concatenate((image, label), axis=0)
    data_dict = DataLoader3D(data, patch_size=[73, 80, 64], final_patch_size=[40, 56, 40], batch_size=1,
                         has_prev_stage=False, pad_mode="constant", pad_sides=None,
                         memmap_mode='r').generate_train_batch()
    image = data_dict['data']
    label = data_dict['seg']
    image, label = tr_transforms(image, label)
    image = np.squeeze(image, 0)
    label = np.squeeze(label, 0)
    label = np.squeeze(label, 0)
    label = self._to_one_hot(label, [0, 1, 2])
    image = np.moveaxis(image, 0, -1)
    label = np.moveaxis(label, 0, -1)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    return image, label

  def _data_augmentation_val(self, image, label):
    image = image.numpy()
    label = label.numpy()
    label = label[np.newaxis, :]
    data = np.concatenate((image, label), axis=0)
    data_dict = DataLoader3D(data, patch_size=[40, 56, 40], final_patch_size=[40, 56, 40], batch_size=1,
                         has_prev_stage=False, pad_mode="constant", pad_sides=None,
                         memmap_mode='r').generate_train_batch()
    image = data_dict['data']
    label = data_dict['seg']
    image, label = val_transforms(image, label)
    image = np.squeeze(image, 0)
    label = np.squeeze(label, 0)
    label = np.squeeze(label, 0)
    label = self._to_one_hot(label, [0, 1, 2])
    image = np.moveaxis(image, 0, -1)
    label = np.moveaxis(label, 0, -1)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    label = tf.convert_to_tensor(label, dtype=tf.float32)
    return image, label


  # One-hot-encoding for seg labels
  def _to_one_hot(self, seg, all_seg_labels=None):
    if all_seg_labels is None:
      all_seg_labels = np.unique(seg)
    result = np.zeros((len(all_seg_labels), *seg.shape), dtype=seg.dtype)
    for i, l in enumerate(all_seg_labels):
      result[i][seg == l] = 1
    return result

  def _parse_train_data(self, data: Dict[str,
                                         Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label_tr(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels

  def _parse_eval_data(self, data: Dict[str,
                                        Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label_val(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels
