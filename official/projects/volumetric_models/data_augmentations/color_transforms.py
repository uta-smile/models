import numpy as np
from official.projects.volumetric_models.data_augmentations.abstract_transforms import AbstractTransform
from official.projects.volumetric_models.data_augmentations.color_augmentations import \
    augment_brightness_multiplicative, augment_contrast, augment_gamma


class BrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
                                                                                self.multiplier_range,
                                                                                self.per_channel)
        return data_dict


class ContrastAugmentationTransform(AbstractTransform):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, data_key="data",
                 p_per_sample=1):
        """
        Augments the contrast of data
        :param contrast_range: range from which to sample a random contrast that is applied to the data. If
        one value is smaller and one is larger than 1, half of the contrast modifiers will be >1 and the other half <1
        (in the inverval that was specified)
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
                                                               contrast_range=self.contrast_range,
                                                               preserve_range=self.preserve_range,
                                                               per_channel=self.per_channel)
        return data_dict


class GammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data", retain_stats=False,
                 p_per_sample=1):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                            self.invert_image,
                                                            per_channel=self.per_channel,
                                                            retain_stats=self.retain_stats)
        return data_dict
