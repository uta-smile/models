import os
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import OrderedDict


class SlimDataLoaderBase(object):
    def __init__(self, data, batch_size, number_of_threads_in_multithreaded=None):
        """
        Slim version of DataLoaderBase (which is now deprecated). Only provides very simple functionality.

        You must derive from this class to implement your own DataLoader. You must overrive self.generate_train_batch()

        If you use our MultiThreadedAugmenter you will need to also set and use number_of_threads_in_multithreaded. See
        multithreaded_dataloading in examples!

        :param data: will be stored in self._data. You can use it to generate your batches in self.generate_train_batch()
        :param batch_size: will be stored in self.batch_size for use in self.generate_train_batch()
        :param number_of_threads_in_multithreaded: will be stored in self.number_of_threads_in_multithreaded.
        None per default. If you wish to iterate over all your training data only once per epoch, you must coordinate
        your Dataloaders and you will need this information
        """
        __metaclass__ = ABCMeta
        self.number_of_threads_in_multithreaded = number_of_threads_in_multithreaded
        self._data = data
        self.batch_size = batch_size
        self.thread_id = 0

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    @abstractmethod
    def generate_train_batch(self):
        '''override this
        Generate your batch from self._data .Make sure you generate the correct batch size (self.BATCH_SIZE)
        '''
        pass


class DataLoader3D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size, has_prev_stage=False, memmap_mode="r",
                 pad_mode="edge", pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 3D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With this strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param random: Sample keys randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param oversample_foreground: half the batch will be forced to contain at least some foreground (equal prob for each of the foreground classes)
        """
        super(DataLoader3D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.final_patch_size = final_patch_size
        self.has_prev_stage = has_prev_stage
        self.patch_size = patch_size

        # need_to_pad denotes by how much we need to pad the data so that if we sample a patch of size final_patch_size
        # (which is what the network will get) these patches will also cover the border of the patients
        self.need_to_pad = (np.array(patch_size) - np.array(final_patch_size)).astype(int)
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.memmap_mode = memmap_mode
        self.num_channels = None
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        if self.has_prev_stage:
            num_seg = 2
        else:
            num_seg = 1

        case_all_data = self._data
        num_color_channels = case_all_data.shape[0] - 1
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape

    def generate_train_batch(self):
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        force_fg = False
        case_all_data = self._data

        need_to_pad = self.need_to_pad
        for d in range(3):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        shape = case_all_data.shape[1:]
        lb_x = - need_to_pad[0] // 2
        ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
        lb_y = - need_to_pad[1] // 2
        ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
        lb_z = - need_to_pad[2] // 2
        ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        bbox_x_ub = bbox_x_lb + self.patch_size[0]
        bbox_y_ub = bbox_y_lb + self.patch_size[1]
        bbox_z_ub = bbox_z_lb + self.patch_size[2]

        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[2], bbox_z_ub)

        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)
        case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                valid_bbox_y_lb:valid_bbox_y_ub,
                                valid_bbox_z_lb:valid_bbox_z_ub])

        data[0] = np.pad(case_all_data[:-1], ((0, 0),
                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                         self.pad_mode, **self.pad_kwargs_data)

        seg[0, 0] = np.pad(case_all_data[-1:], ((0, 0),
                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                           'constant', **{'constant_values': -1})

        return {'data': data, 'seg': seg}


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, patch_size, final_patch_size, batch_size,
                 memmap_mode="r", pseudo_3d_slices=1, pad_mode="edge",
                 pad_kwargs_data=None, pad_sides=None):
        """
        This is the basic data loader for 2D networks. It uses preprocessed data as produced by my (Fabian) preprocessing.
        You can load the data with load_dataset(folder) where folder is the folder where the npz files are located. If there
        are only npz files present in that folder, the data loader will unpack them on the fly. This may take a while
        and increase CPU usage. Therefore, I advise you to call unpack_dataset(folder) first, which will unpack all npz
        to npy. Don't forget to call delete_npy(folder) after you are done with training?
        Why all the hassle? Well the decathlon dataset is huge. Using npy for everything will consume >1 TB and that is uncool
        given that I (Fabian) will have to store that permanently on /datasets and my local computer. With htis strategy all
        data is stored in a compressed format (factor 10 smaller) and only unpacked when needed.
        :param data: get this with load_dataset(folder, stage=0). Plug the return value in here and you are g2g (good to go)
        :param patch_size: what patch size will this data loader return? it is common practice to first load larger
        patches so that a central crop after data augmentation can be done to reduce border artifacts. If unsure, use
        get_patch_size() from data_augmentation.default_data_augmentation
        :param final_patch_size: what will the patch finally be cropped to (after data augmentation)? this is the patch
        size that goes into your network. We need this here because we will pad patients in here so that patches at the
        border of patients are sampled properly
        :param batch_size:
        :param num_batches: how many batches will the data loader produce before stopping? None=endless
        :param seed:
        :param stage: ignore this (Fabian only)
        :param transpose: ignore this
        :param random: sample randomly; CAREFUL! non-random sampling requires batch_size=1, otherwise you will iterate batch_size times over the dataset
        :param pseudo_3d_slices: 7 = 3 below and 3 above the center slice
        """
        super(DataLoader2D, self).__init__(data, batch_size, None)
        if pad_kwargs_data is None:
            pad_kwargs_data = OrderedDict()
        self.pad_kwargs_data = pad_kwargs_data
        self.pad_mode = pad_mode
        self.pseudo_3d_slices = pseudo_3d_slices
        self.final_patch_size = final_patch_size
        self.patch_size = patch_size

        self.need_to_pad = np.array(patch_size) - np.array(final_patch_size)
        self.memmap_mode = memmap_mode
        if pad_sides is not None:
            if not isinstance(pad_sides, np.ndarray):
                pad_sides = np.array(pad_sides)
            self.need_to_pad += pad_sides
        self.pad_sides = pad_sides
        self.data_shape, self.seg_shape = self.determine_shapes()

    def determine_shapes(self):
        num_seg = 1
        case_all_data = self._data
        num_color_channels = case_all_data.shape[0] - num_seg
        data_shape = (self.batch_size, num_color_channels, *self.patch_size)
        seg_shape = (self.batch_size, num_seg, *self.patch_size)
        return data_shape, seg_shape


    def generate_train_batch(self):
        data = np.zeros(self.data_shape, dtype=np.float32)
        seg = np.zeros(self.seg_shape, dtype=np.float32)

        force_fg = False
        case_all_data = self._data

        # first select a slice. This can be either random (no force fg) or guaranteed to contain some class
        if not force_fg:
            random_slice = np.random.choice(case_all_data.shape[1])
            selected_class = None


        # now crop case_all_data to contain just the slice of interest. If we want additional slice above and
        # below the current slice, here is where we get them. We stack those as additional color channels
        if self.pseudo_3d_slices == 1:
            case_all_data = case_all_data[:, random_slice]
            # case_all_data = case_all_data[:, 20]



        # case all data should now be (c, x, y)
        assert len(case_all_data.shape) == 3

        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint

        need_to_pad = self.need_to_pad
        for d in range(2):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]

        shape = case_all_data.shape[1:]
        lb_x = - need_to_pad[0] // 2
        ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
        lb_y = - need_to_pad[1] // 2
        ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg or selected_class is None:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)


        bbox_x_ub = bbox_x_lb + self.patch_size[0]
        bbox_y_ub = bbox_y_lb + self.patch_size[1]

        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)

        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)

        case_all_data = case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                        valid_bbox_y_lb:valid_bbox_y_ub]

        case_all_data_donly = np.pad(case_all_data[:-1], ((0, 0),
                                                          (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                          (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                     self.pad_mode, **self.pad_kwargs_data)

        case_all_data_segonly = np.pad(case_all_data[-1:], ((0, 0),
                                                            (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                            (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0))),
                                       'constant', **{'constant_values': -1})

        data[0] = case_all_data_donly
        seg[0, 0] = case_all_data_segonly

        # print(data.shape, seg.shape)
        return {'data': data, 'seg': seg}

        
if __name__ == "__main__":
    pass
