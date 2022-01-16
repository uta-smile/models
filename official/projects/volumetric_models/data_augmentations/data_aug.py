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
def tf_tr_transforms(self):

    da = tf.keras.layers.Sequential(
        [
            tf.keras.layers.Input(
                type_spec=TFDAData.Spec(None, tf.TensorSpec(None), tf.TensorSpec(None))
            ),
            SpatialTransform(
                patch_size=self.patch_size,
                patch_center_dist_from_border=nan,
                do_elastic_deform=self.data_aug_param.get("do_elastic"),
                alpha=self.data_aug_param.get("elastic_deform_alpha"),
                sigma=self.data_aug_param.get("elastic_deform_sigma"),
                do_rotation=self.data_aug_param.get("do_rotation"),
                angle_x=self.data_aug_param.get("rotation_x"),
                angle_y=self.data_aug_param.get("rotation_y"),
                angle_z=self.data_aug_param.get("rotation_z"),
                p_rot_per_axis=self.data_aug_param.get("rotation_p_per_axis"),
                do_scale=self.data_aug_param.get("do_scaling"),
                scale=self.data_aug_param.get("scale_range"),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=3,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=1,
                random_crop=self.data_aug_param.get("random_crop"),
                p_el_per_sample=self.data_aug_param.get("p_eldef"),
                p_scale_per_sample=self.data_aug_param.get("p_scale"),
                p_rot_per_sample=self.data_aug_param.get("p_rot"),
                independent_scale_for_each_axis=self.data_aug_param.get(
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
            GammaTransform(
                self.data_aug_param.get("gamma_range"),
                True,
                True,
                retain_stats=self.data_aug_param.get("gamma_retain_stats"),
                p_per_sample=0.1,
            ),
            GammaTransform(
                self.data_aug_param.get("gamma_range"),
                False,
                True,
                retain_stats=self.data_aug_param.get("gamma_retain_stats"),
                p_per_sample=self.data_aug_param.get("p_gamma"),
            ),
            MirrorTransform(self.data_aug_param.get("mirror_axes")),
            MaskTransform(tf.constant([[0, 0]]), mask_idx_in_seg=0, set_outside_to=0.0),
            RemoveLabelTransform(-1, 0),
            OneHotTransform(tf.nest.map_structure(float, self._jsn["labels"].keys())),
        ]
    )
    da.compile()
    return da


@tf.function
def tf_val_transforms(self):
    da = tf.keras.layers.Sequential(
        [
            tf.keras.layers.Input(
                type_spec=TFDAData.Spec(None, tf.TensorSpec(None), tf.TensorSpec(None))
            ),
            RemoveLabelTransform(-1, 0),
            OneHotTransform(tf.nest.map_structure(float, self._jsn["labels"].keys())),
        ]
    )
    da.compile()
    return da
