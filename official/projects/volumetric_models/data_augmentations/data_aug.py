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


# Tensorflow
import tensorflow as tf

# Local
from tfda.augmentations.utils import to_one_hot
from tfda.defs import TFDAData, nan
from tfda.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from tfda.transforms.custom_transforms import MaskTransform, OneHotTransform
from tfda.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from tfda.transforms.resample_transforms import SimulateLowResolutionTransform
from tfda.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from tfda.transforms.utility_transforms import RemoveLabelTransform


@tf.function
def tf_tr_transforms(
    images: tf.Tensor,
    segs: tf.Tensor,
    dator,
    border_val_seg=-1,
    seeds_train=None,
    seeds_val=None,
    order_seg=1,
    order_data=3,
    deep_supervision_scales=None,
    soft_ds=False,
    classes=None,
    pin_memory=True,
    regions=None,
    use_nondetMultiThreadedAugmenter: bool = False,
):
    params = dator.data_aug_param
    images = tf.transpose(images, (0, 4, 1, 2, 3))
    segs = tf.transpose(segs, (0, 4, 1, 2, 3))

    da = tf.keras.layers.Sequential(
        [
            tf.keras.layers.Input(
                type_spec=TFDAData.Spec(None, tf.TensorSpec(None), tf.TensorSpec(None))
            ),
            SpatialTransform(
                dator.patch_size,
                patch_center_dist_from_border=nan,
                do_elastic_deform=params.get("do_elastic"),
                alpha=params.get("elastic_deform_alpha"),
                sigma=params.get("elastic_deform_sigma"),
                do_rotation=params.get("do_rotation"),
                angle_x=params.get("rotation_x"),
                angle_y=params.get("rotation_y"),
                angle_z=params.get("rotation_z"),
                p_rot_per_axis=params.get("rotation_p_per_axis"),
                do_scale=params.get("do_scaling"),
                scale=params.get("scale_range"),
                border_mode_data=params.get("border_mode_data"),
                border_cval_data=0,
                order_data=order_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_seg,
                random_crop=params.get("random_crop"),
                p_el_per_sample=params.get("p_eldef"),
                p_scale_per_sample=params.get("p_scale"),
                p_rot_per_sample=params.get("p_rot"),
                independent_scale_for_each_axis=params.get(
                    "independent_scale_factor_for_each_axis"
                ),
            ),
            GaussianNoiseTransform(
                data_key="data", label_key="seg", p_per_channel=0.01
            ),
            GaussianBlurTransform(
                (0.5, 1.0),
                different_sigma_per_channel=True,
                p_per_sample=0.2,
                p_per_channel=0.5,
            ),
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.75, 1.25), p_per_sample=0.15
            ),
            ContrastAugmentationTransform(p_per_sample=0.15),
            SimulateLowResolutionTransform(
                zoom_range=(0.5, 1),
                per_channel=True,
                p_per_channel=0.5,
                order_downsample=0,
                order_upsample=3,
                p_per_sample=0.25,
            ),
            GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1),
            GammaTransform(
                (0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3
            ),
            MirrorTransform((0, 1, 2)),
            MaskTransform(tf.constant([[0, 0]]), mask_idx_in_seg=0, set_outside_to=0.0),
            RemoveLabelTransform(-1, 0),
            OneHotTransform(),
        ]
    )

    da.compile()

    data_dict = da(TFDAData(data=images, seg=segs))

    images = data_dict.data
    segs = data_dict.seg

    return images, segs


@tf.function
def tf_val_transforms(
    images,
    segs,
    dator,
    border_val_seg=-1,
    seeds_train=None,
    seeds_val=None,
    order_seg=1,
    order_data=3,
    deep_supervision_scales=None,
    soft_ds=False,
    classes=None,
    pin_memory=True,
    regions=None,
    use_nondetMultiThreadedAugmenter: bool = False,
):
    params = dator.data_aug_param
    images = tf.transpose(images, (0, 4, 1, 2, 3))
    segs = tf.transpose(segs, (0, 4, 1, 2, 3))

    da = tf.keras.layers.Sequential(
        [
            tf.keras.layers.Input(
                type_spec=TFDAData.Spec(None, tf.TensorSpec(None), tf.TensorSpec(None))
            ),
            RemoveLabelTransform(-1, 0),
            OneHotTransform(),
        ]
    )
    da.compile()
    data_dict = da(TFDAData(data=images, seg=segs))
    images = data_dict.data
    segs = data_dict.seg
    # tf.print(tf.shape(data_dict['data']), tf.shape(data_dict['seg']))
    return images, segs
