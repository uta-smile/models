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
from official.projects.volumetric_models.data_augmentations.data_aug import tf_tr_transforms, tf_val_transforms

from tfda.transforms.spatial_transforms import SpatialTransform, MirrorTransform, SpatialTransform2D, MirrorTransform2D
from tfda.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform, GaussianBlurTransform2D
from tfda.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from tfda.transforms.custom_transforms import MaskTransform, OneHotTransform, Convert3DTo2DTransform, Convert2DTo3DTransform, OneHotTransform2D
from tfda.transforms.utility_transforms import RemoveLabelTransform
from tfda.transforms.resample_transforms import SimulateLowResolutionTransform, SimulateLowResolutionTransform2D
from tfda.defs import TFDAData, TFDADefault3DParams, DTFT, TFbF, TFbT, nan, pi
from tfda.data_processing_utils import get_batch_size, update_tf_channel


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
              label_dtype: str = 'int32',
              pkl = None,
              jsn = None):
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
    self._pkl = pkl
    self._jsn = jsn
    self.process_plans()
    self.setup_DA_params()
    # self.tr_da = tf_tr_transforms(self)
    # self.val_da = tf_val_transforms(self)

  def _prepare_image_and_label(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = tf.io.decode_raw(data[self._image_field_key],
                             tf.as_dtype(tf.float32))
    label = tf.io.decode_raw(data[self._label_field_key],
                             tf.as_dtype(self._label_dtype))
    image_size = list(self._input_size) + [self._num_channels]
    image = tf.reshape(image, image_size)
    label_size = list(self._input_size) + [self._num_classes]
    label = tf.reshape(label, label_size)

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

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

    image, label = self._data_augmentation_tr(image, label)

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

    image, label = self._data_augmentation_val(image, label)

    image.set_shape(self._input_size+[self._num_channels])
    label.set_shape(self._input_size+[self._num_classes])

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._label_dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _data_augmentation_tr(self, image, label):
    image, label = process_batch2d(image, label[tf.newaxis,], self.basic_generator_patch_size, self.patch_size, 1)
    image, label = self.tf_tr_transforms(image, label)
    return image[0], label[0]

  def _data_augmentation_val(self, image, label):
    image, label = process_batch2d(image, label[tf.newaxis,], self.patch_size, self.patch_size, 1)
    image, label = self.tf_val_transforms(image, label)
    return image[0], label[0]

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

  def process_plans(self):
    #  TODO here, we do not consider cascade. So, we always pick the first stage plan
    self.stage = list(self._pkl['plans_per_stage'].keys())[0]
    stage_plans = self._pkl['plans_per_stage'][self.stage]

    self.batch_size = stage_plans['batch_size']
    self.patch_size = stage_plans['patch_size']
    self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
    self.pad_all_sides = None
    self.use_mask_for_norm = self._pkl['use_mask_for_norm'][0]
    if len(self.patch_size) == 2:
      self.threeD = False
    else:
      self.threeD = True  # we only consider 2 circumstances: 3D and 2D

  def setup_DA_params(self):
    if self.threeD:
      rotation_x = (
          -30.0 / 360 * 2.0 * pi,
          30.0 / 360 * 2.0 * pi,
      )
      rotation_y = (
          -30.0 / 360 * 2.0 * pi,
          30.0 / 360 * 2.0 * pi,
      )
      rotation_z = (
          -30.0 / 360 * 2.0 * pi,
          30.0 / 360 * 2.0 * pi,
      )
      if self.do_dummy_2D_aug:
          dummy_2D = TFbT
          # print("Using dummy2d data augmentation")
          elastic_deform_alpha = (0.0, 200.0)
          elastic_deform_sigma = (9.0, 13.0)
          rotation_x = (
              -180.0 / 360 * 2.0 * pi,
              180.0 / 360 * 2.0 * pi,
          )
          self.data_aug_param = TFDADefault3DParams(
              rotation_x=rotation_x,
              rotation_y=rotation_y,
              rotation_z=rotation_z,
              dummy_2D=dummy_2D,
              elastic_deform_alpha=elastic_deform_alpha,
              elastic_deform_sigma=elastic_deform_sigma,
              scale_range=(0.7, 1.4),
              do_elastic=TFbF,
              selected_seg_channels=[0],
              patch_size_for_spatial_transform=self.patch_size,
              num_cached_per_thread=2,
              mask_was_used_for_normalization=self.use_mask_for_norm,
          )
      else:
          self.data_aug_param = TFDADefault3DParams(
              rotation_x=rotation_x,
              rotation_y=rotation_y,
              rotation_z=rotation_z,
              scale_range=(0.7, 1.4),
              do_elastic=TFbF,
              selected_seg_channels=[0],
              patch_size_for_spatial_transform=self.patch_size,
              num_cached_per_thread=2,
              mask_was_used_for_normalization=self.use_mask_for_norm,
          )
    else:
      self.do_dummy_2D_aug = TFbF
      if tf.maximum(self.patch_size) / tf.minimum(self.patch_size) > 1.5:
          rotation_x = (
              -15.0 / 360 * 2.0 * pi,
              15.0 / 360 * 2.0 * pi,
          )
      else:
          rotation_x = (
              -180.0 / 360 * 2.0 * pi,
              180.0 / 360 * 2.0 * pi,
          )
      elastic_deform_alpha = (0.0, 200.0)
      elastic_deform_sigma = (9.0, 13.0)
      rotation_y = (
          -0.0 / 360 * 2.0 * pi,
          0.0 / 360 * 2.0 * pi,
      )
      rotation_z = (
          -0.0 / 360 * 2.0 * pi,
          0.0 / 360 * 2.0 * pi,
      )
      dummy_2D = TFbF
      mirror_axes = (
          0,
          1,
      )
      self.data_aug_param = TFDADefault3DParams(
          rotation_x=rotation_x,
          rotation_y=rotation_y,
          rotation_z=rotation_z,
          elastic_deform_alpha=elastic_deform_alpha,
          elastic_deform_sigma=elastic_deform_sigma,
          dummy_2D=dummy_2D,
          mirror_axes=mirror_axes,
          mask_was_used_for_normalization=self.use_mask_for_norm,
          scale_range=(0.7, 1.4),
          do_elastic=TFbF,
          selected_seg_channels=[0],
          patch_size_for_spatial_transform=self.patch_size,
          num_cached_per_thread=2,
            )

    if self.do_dummy_2D_aug:
        self.basic_generator_patch_size = get_batch_size(
            self.patch_size[1:],
            self.data_aug_param["rotation_x"],
            self.data_aug_param["rotation_y"],
            self.data_aug_param["rotation_z"],
            (0.85, 1.25),
        )
        self.basic_generator_patch_size = tf.concat([tf.convert_to_tensor([self.patch_size[0]], tf.int32), self.basic_generator_patch_size], axis=0)

    else:
        self.basic_generator_patch_size = get_batch_size(
            self.patch_size,
            self.data_aug_param["rotation_x"],
            self.data_aug_param["rotation_y"],
            self.data_aug_param["rotation_z"],
            (0.85, 1.25),
        )
    self.basic_generator_patch_size = tf.cast(
        self.basic_generator_patch_size, tf.int64
    )


#   @tf.function
  def tf_tr_transforms(self, images, segs):
    #   tf.config.run_functions_eagerly(True)
    data_dict = TFDAData(data=images, seg=segs)
    #   tf.print(tf.shape(data_dict.data), tf.shape(data_dict.seg))
    patch_size = self.patch_size[1:]
    #   tf.print(tf.shape(data_dict.data), tf.shape(data_dict.seg))
    #   tf.print(patch_size)
    data_dict = SpatialTransform2D(
      patch_size=patch_size, patch_center_dist_from_border=nan,
      do_elastic_deform=self.data_aug_param.get('do_elastic'), alpha=self.data_aug_param.get('elastic_deform_alpha'),
      sigma=self.data_aug_param.get('elastic_deform_sigma'),
      do_rotation=self.data_aug_param.get("do_rotation"), angle_x=self.data_aug_param.get("rotation_x"),
      angle_y=self.data_aug_param.get("rotation_y"),
      angle_z=self.data_aug_param.get("rotation_z"), p_rot_per_axis=self.data_aug_param.get("rotation_p_per_axis"),
      do_scale=self.data_aug_param.get("do_scaling"), scale=self.data_aug_param.get("scale_range"), border_mode_data='constant', border_cval_data=0,
      order_data=3, border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
      random_crop=self.data_aug_param.get("random_crop"), p_el_per_sample=self.data_aug_param.get("p_eldef"), p_scale_per_sample=self.data_aug_param.get("p_scale"), p_rot_per_sample=self.data_aug_param.get("p_rot"),
      independent_scale_for_each_axis=self.data_aug_param.get("independent_scale_factor_for_each_axis")
    )(data_dict)
    #   tf.config.run_functions_eagerly(False)

    data_dict = GaussianNoiseTransform(data_key="data", label_key="seg", p_per_channel=0.01)(data_dict)
    data_dict = GaussianBlurTransform2D((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                        p_per_channel=0.5)(data_dict)
    data_dict = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)(data_dict)
    data_dict = ContrastAugmentationTransform(p_per_sample=0.15)(data_dict)

    data_dict = SimulateLowResolutionTransform2D(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                                 order_downsample=0, order_upsample=3, p_per_sample=0.25)(data_dict)

    data_dict = GammaTransform(self.data_aug_param.get("gamma_range"), True, True, retain_stats=self.data_aug_param.get("gamma_retain_stats"), p_per_sample=0.1)(data_dict)
    data_dict = GammaTransform(self.data_aug_param.get("gamma_range"), False, True, retain_stats=self.data_aug_param.get("gamma_retain_stats"), p_per_sample=self.data_aug_param.get("p_gamma"))(data_dict)
    data_dict = MirrorTransform2D(self.data_aug_param.get("mirror_axes"))(data_dict)
    data_dict = MaskTransform(tf.constant([[0, 0]]), mask_idx_in_seg=0, set_outside_to=0.0)(data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(data_dict)
    data_dict = OneHotTransform2D(tuple([float(key) for key in self._jsn['labels'].keys()]))(data_dict)
    #   tf.print('tr', tf.shape(data_dict.data))
    return data_dict.data, data_dict.seg


#   @tf.function
  def tf_val_transforms(self, images, segs):
    data_dict = RemoveLabelTransform(-1, 0)(TFDAData(data=images, seg=segs))
    data_dict = OneHotTransform2D(tuple([float(key) for key in self._jsn['labels'].keys()]))(data_dict)
    #   tf.print('val', tf.shape(data_dict.data))
    return data_dict.data, data_dict.seg


@tf.function
def update_need_to_pad(
    need_to_pad, d, basic_generator_patch_size, case_all_data
):
    need_to_pad_d = (
        basic_generator_patch_size[d]
        - tf.shape(case_all_data, out_type=tf.int64)[d + 1]
    )
    return tf.cond(
        tf.less(
            need_to_pad[d] + tf.shape(case_all_data, out_type=tf.int64)[d + 1],
            basic_generator_patch_size[d],
        ),
        lambda: need_to_pad_d,
        lambda: need_to_pad[d],
    )


@tf.function
def not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z):
    bbox_x_lb = tf.random.uniform(
        [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
    )
    bbox_y_lb = tf.random.uniform(
        [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
    )
    bbox_z_lb = tf.random.uniform(
        [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
    )
    return bbox_x_lb, bbox_y_lb, bbox_z_lb


@tf.function
def process_batch(
    image,
    label,
    basic_generator_patch_size,
    patch_size,
):
    zero = tf.constant(0, dtype=tf.int64)
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    case_all_data = tf.concat([image, label], axis=0)
    basic_generator_patch_size = tf.cast(
        basic_generator_patch_size, dtype=tf.int64
    )
    patch_size = tf.cast(patch_size, dtype=tf.int64)
    need_to_pad = basic_generator_patch_size - patch_size
    need_to_pad = tf.map_fn(
        lambda d: update_need_to_pad(
            need_to_pad, d, basic_generator_patch_size, case_all_data
        ),
        elems=tf.range(3, dtype=tf.int64),
    )
    need_to_pad = tf.cast(need_to_pad, tf.int64)
    shape = tf.shape(case_all_data, out_type=tf.int64)[1:]
    lb_x = -need_to_pad[0] // 2
    ub_x = (
        shape[0]
        + need_to_pad[0] // 2
        + need_to_pad[0] % 2
        - basic_generator_patch_size[0]
    )
    lb_y = -need_to_pad[1] // 2
    ub_y = (
        shape[1]
        + need_to_pad[1] // 2
        + need_to_pad[1] % 2
        - basic_generator_patch_size[1]
    )
    lb_z = -need_to_pad[2] // 2
    ub_z = (
        shape[2]
        + need_to_pad[2] // 2
        + need_to_pad[2] % 2
        - basic_generator_patch_size[2]
    )

    bbox_x_lb, bbox_y_lb, bbox_z_lb = not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z)

    bbox_x_ub = bbox_x_lb + basic_generator_patch_size[0]
    bbox_y_ub = bbox_y_lb + basic_generator_patch_size[1]
    bbox_z_ub = bbox_z_lb + basic_generator_patch_size[2]

    valid_bbox_x_lb = tf.maximum(zero, bbox_x_lb)
    valid_bbox_x_ub = tf.minimum(shape[0], bbox_x_ub)
    valid_bbox_y_lb = tf.maximum(zero, bbox_y_lb)
    valid_bbox_y_ub = tf.minimum(shape[1], bbox_y_ub)
    valid_bbox_z_lb = tf.maximum(zero, bbox_z_lb)
    valid_bbox_z_ub = tf.minimum(shape[2], bbox_z_ub)

    case_all_data = tf.identity(
        case_all_data[
            :,
            valid_bbox_x_lb:valid_bbox_x_ub,
            valid_bbox_y_lb:valid_bbox_y_ub,
            valid_bbox_z_lb:valid_bbox_z_ub,
        ]
    )

    img = tf.pad(
        case_all_data[:-1],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
    )
    seg = tf.pad(
        case_all_data[-1:],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
        constant_values=-1,
    )
    return img[tf.newaxis,], seg[tf.newaxis,]

@tf.function
def process_batch2d(
    image,
    label,
    basic_generator_patch_size,
    patch_size,
    pseud_3d_slices,
    ):
    zero = tf.constant(0, dtype=tf.int64)
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    case_all_data = tf.concat([image, label], axis=0)
    case_all_data = tf.cond(tf.equal(tf.rank(case_all_data), 3), lambda: case_all_data[:, tf.newaxis], lambda: case_all_data)
    random_slice = tf.random.uniform([], minval=0, maxval=tf.shape(case_all_data)[1], dtype=tf.int32)
    case_all_data = tf.cond(tf.equal(pseud_3d_slices, 1), lambda: case_all_data[:, random_slice], lambda: process_pseud_3d_slices(case_all_data, random_slice, pseud_3d_slices))
    basic_generator_patch_size = tf.cast(basic_generator_patch_size, dtype=tf.int64)
    patch_size = tf.cast(patch_size, dtype=tf.int64)
    need_to_pad = basic_generator_patch_size - patch_size
    need_to_pad = tf.map_fn(
        lambda d: update_need_to_pad(
            need_to_pad, d, basic_generator_patch_size, case_all_data
        ),
        elems=tf.range(2, dtype=tf.int64),
    )
    need_to_pad = tf.cast(need_to_pad, tf.int64)
    shape = tf.shape(case_all_data)[1:]
    lb_x = -need_to_pad[0] // 2
    ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - basic_generator_patch_size[0]
    lb_y = -need_to_pad[1] // 2
    ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - basic_generator_patch_size[1]

    #TODO force_fg = False
    bbox_x_lb = tf.random.uniform([], minval=lb_x, maxval=ub_x+1, dtype=tf.int64)
    bbox_y_lb = tf.random.uniform([], minval=lb_y, maxval=ub_y+1, dtype=tf.int64)

    bbox_x_ub = bbox_x_lb + basic_generator_patch_size[0]
    bbox_y_ub = bbox_y_lb + basic_generator_patch_size[1]
    valid_bbox_x_lb = tf.maximum(zero, bbox_x_lb)
    valid_bbox_x_ub = tf.minimum(shape[0], bbox_x_ub)
    valid_bbox_y_lb = tf.maximum(zero, bbox_y_lb)
    valid_bbox_y_ub = tf.minimum(shape[1], bbox_y_ub)
    case_all_data = case_all_data[:, valid_bbox_x_lb: valid_bbox_x_ub,
                                valid_bbox_y_lb: valid_bbox_y_ub]
    case_all_data_donly = tf.pad(case_all_data[:-1], [[0, 0],
                                                    [-tf.minimum(zero, bbox_x_lb), tf.maximum(bbox_x_ub - shape[0], zero)],
                                                    [-tf.minimum(zero, bbox_y_lb), tf.maximum(bbox_y_ub - shape[1], zero)]])
    case_all_data_segonly = tf.pad(case_all_data[-1:], [[0, 0],
                                                        [-tf.minimum(zero, bbox_x_lb), tf.maximum(bbox_x_ub - shape[0], zero)],
                                                        [-tf.minimum(zero, bbox_y_lb), tf.maximum(bbox_y_ub - shape[1], zero)]],
                                constant_values=-1)

    return case_all_data_donly[tf.newaxis,], case_all_data_segonly[tf.newaxis,]

@tf.function
def process_pseud_3d_slices(case_all_data, random_slice, pseud_3d_slices):
    zero = tf.constant(0, dtype=tf.int64)
    mn = random_slice - (pseud_3d_slices - 1) // 2
    mx = random_slice + (pseud_3d_slices - 1) // 2 + 1
    valid_mn = tf.maximum(mn, zero)
    valid_mx = tf.minimum(mx, tf.shape(case_all_data)[1])
    case_all_seg = case_all_data[-1:]
    case_all_data = case_all_data[:-1]
    case_all_data = case_all_data[:, valid_mn: valid_mx]
    case_all_seg = case_all_seg[:, random_slice]
    need_to_pad_below = valid_mn - mn
    need_to_pad_above = mx - valid_mx
    case_all_data = tf.cond(tf.greater(need_to_pad_below, zero), lambda: process_need_to_pad_below(case_all_data, need_to_pad_below), lambda: case_all_data)
    case_all_data = tf.cond(tf.greater(need_to_pad_above, zero), lambda: process_need_to_pad_above(case_all_data, need_to_pad_above), lambda: case_all_data)
    case_all_data = tf.reshape(case_all_data, (-1, tf.shape(case_all_data)[-2], tf.shape(case_all_data)[-1]))
    case_all_data = tf.concat([case_all_data, case_all_seg], axis=0)
    return case_all_data

@tf.function
def process_need_to_pad_below(case_all_data, need_to_pad_below):
    shp_for_pad = tf.shape(case_all_data)
    shp_for_pad_1 = need_to_pad_below
    shp_for_pad = update_tf_channel(shp_for_pad, 1, shp_for_pad_1)
    case_all_data = tf.concat([tf.zeros(shp_for_pad), case_all_data], axis=1)
    return case_all_data

@tf.function
def process_need_to_pad_above(case_all_data, need_to_pad_above):
    shp_for_pad = tf.shape(case_all_data)
    shp_for_pad_1 = need_to_pad_above
    shp_for_pad = update_tf_channel(shp_for_pad, 1, shp_for_pad_1)
    case_all_data = tf.concat([case_all_data, tf.zeros(shp_for_pad)], axis=1)
