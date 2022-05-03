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

from official.projects.volumetric_models.data_augmentations.dataset_loading import DataLoader2D
from official.projects.volumetric_models.data_augmentations.spatial_transforms import SpatialTransform, MirrorTransform
from official.projects.volumetric_models.data_augmentations.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from official.projects.volumetric_models.data_augmentations.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from official.projects.volumetric_models.data_augmentations.resample_transforms import SimulateLowResolutionTransform
from official.projects.volumetric_models.data_augmentations.custom_transforms import MaskTransform
from official.projects.volumetric_models.data_augmentations.utility_transforms import RemoveLabelTransform


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
               input_size: Sequence[int] = [56, 40],
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
    data_dict = DataLoader2D(data, patch_size=[65, 65], final_patch_size=[56, 40], batch_size=1,
                             pad_mode="constant", pad_sides=None, memmap_mode='r').generate_train_batch()
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
    data_dict = DataLoader2D(data, patch_size=[56, 40], final_patch_size=[56, 40], batch_size=1,
                             pad_mode="constant", pad_sides=None, memmap_mode='r').generate_train_batch()
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


def tr_transforms(image, label):
    data_dict = {'data': image, 'seg': label}
    data_dict = SpatialTransform(patch_size=[56, 40], patch_center_dist_from_border=None,
                                 do_elastic_deform=False, alpha=(0.0, 200.0), sigma=(9.0, 13.0),
                                 do_rotation=True, angle_x=(-0.5235987755982988, 0.5235987755982988),
                                 angle_y=(-0.0, 0.0), angle_z=(-0.0, 0.0), p_rot_per_axis=1,
                                 do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0,
                                 order_data=3, border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
                                 random_crop=False, p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                                 independent_scale_for_each_axis=False)(**data_dict)
    data_dict = GaussianNoiseTransform(p_per_sample=0.1)(**data_dict)
    data_dict = GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                      p_per_channel=0.5)(**data_dict)
    data_dict = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)(**data_dict)
    data_dict = ContrastAugmentationTransform(p_per_sample=0.15)(**data_dict)
    data_dict = SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                               order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                               ignore_axes=None)(**data_dict)
    data_dict = GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)(**data_dict)
    data_dict = GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)(**data_dict)
    data_dict = MirrorTransform((0, 1))(**data_dict)
    data_dict = MaskTransform([(0, False)], mask_idx_in_seg=0, set_outside_to=0)(**data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    return image, label


def val_transforms(image, label):
    data_dict = {'data': image, 'seg': label}

    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    return image, label