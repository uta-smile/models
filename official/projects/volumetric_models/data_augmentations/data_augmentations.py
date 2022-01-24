import numpy as np
from official.projects.volumetric_models.data_augmentations.spatial_transforms import SpatialTransform, MirrorTransform
from official.projects.volumetric_models.data_augmentations.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from official.projects.volumetric_models.data_augmentations.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from official.projects.volumetric_models.data_augmentations.resample_transforms import SimulateLowResolutionTransform
from official.projects.volumetric_models.data_augmentations.custom_transforms import MaskTransform
from official.projects.volumetric_models.data_augmentations.utility_transforms import RemoveLabelTransform, RenameTransform


def tr_transforms(image, label):
    data_dict = {'data': image, 'seg': label}
    # print(data_dict)

    data_dict = SpatialTransform(patch_size=[40, 56, 40], patch_center_dist_from_border=None,
                                 do_elastic_deform=False, alpha=(0.0, 900.0), sigma=(9.0, 13.0),
                                 do_rotation=True, angle_x=(-0.5235987755982988, 0.5235987755982988),
                                 angle_y=(-0.5235987755982988, 0.5235987755982988),
                                 angle_z=(-0.5235987755982988, 0.5235987755982988), p_rot_per_axis=1,
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
    data_dict = MirrorTransform((0, 1, 2))(**data_dict)
    data_dict = MaskTransform([(0, False)], mask_idx_in_seg=0, set_outside_to=0)(**data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    # print("after transforms:", image.shape, label.shape)
    return image, label


def val_transforms(image, label):
    data_dict = {'data': image, 'seg': label}

    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    return image, label


if __name__ == "__main__":
    pass
