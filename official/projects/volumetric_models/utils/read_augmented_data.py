import os
from pickle import FALSE
from unittest import registerResult
import tensorflow as tf

from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.projects.volumetric_models.dataloaders_tf import segmentation_input_3d_t6 as segmentation_input_3d

def read_tfrecord(tfrecord_file):

    # # task 002
    # input_size = [80, 192, 160]
    # num_classes = 2
    # num_channels = 1

    # # task 003
    # input_size = [128, 128, 128]
    # num_classes = 3
    # num_channels = 1

    # # task 005
    # input_size = [20, 320, 256]
    # num_classes = 3
    # num_channels = 2

    # task 006
    input_size = [80, 192, 160]
    num_classes = 2
    num_channels = 1

    # # task 007
    # input_size = [40, 224, 224]
    # num_classes = 3
    # num_channels = 1

    image_shape_key = 'image_shape'
    label_shape_key  = 'label_shape'
    print("read tfrecord")
    params = cfg.DataConfig(input_path=tfrecord_file, global_batch_size=2, is_training=False)

    decoder = segmentation_input_3d.Decoder()

    parser = segmentation_input_3d.Parser(
        input_size=input_size,
        num_classes=num_classes,
        num_channels=num_channels,
        image_shape_key=image_shape_key,
        label_shape_key=label_shape_key,
        dtype='float32',
        label_dtype='float32')

    reader = input_reader.InputReader(
        params,
        dataset_fn=dataset_fn.pick_dataset_fn('tfrecord'),
        decoder_fn=decoder.decode,
        parser_fn=parser.parse_fn(params.is_training))

    dataset = reader.read()
    iterator = iter(dataset)
    image, labels = next(iterator)
    print("after parse:", type(image), type(labels))
    print("after parse:", type(image.numpy()), type(labels.numpy()))
    print("after parse:", type(image.numpy()[0, 0, 0, 0, 0]), type(labels.numpy()[0, 0, 0, 0, 0]))


    # Checks image shape.
    print("after parse:", list(image.numpy().shape))  # [2, input_size[0], input_size[1], input_size[2], num_channels]
    print("after parse:", list(labels.numpy().shape))  # [2, input_size[0], input_size[1], input_size[2], num_classes]

    for i in range(10):
        image, labels = next(iterator)
        nan_image = tf.math.reduce_any(tf.math.is_nan(image))
        if nan_image:
            print("!!!!!!!!!!!!!!! nan_image")
        nan_labels = tf.math.reduce_any(tf.math.is_nan(labels))
        if nan_labels:
            print("!!!!!!!!!!!!!!! nan_labels")
    
        # Checks image shape.
        print(i, "after parse image:", list(image.numpy().shape))
        print("after parse label:", list(labels.numpy().shape))

    # image = image.numpy()
    # label = labels.numpy()
    # print(image.shape, label.shape)

    # import numpy as np
    # np.save("/home/fengtong/models/official/projects/volumetric_models/visulization/img_channel_0.npy", image[0][:, :, :, 0])
    # np.save("/home/fengtong/models/official/projects/volumetric_models/visulization/lb_class_1.npy", label[0][:, :, :, 1])
    # np.save("/home/fengtong/models/official/projects/volumetric_models/visulization/lb_class_2.npy", label[0][:, :, :, 2])


    # import SimpleITK as sitk
    
    # def numpy_to_niigz(np_data, nii_path):
    #     img = sitk.GetImageFromArray(np_data)
    #     # img = img.astype(np.int16)
    #     sitk.WriteImage(img, nii_path)
    
    # numpy_to_niigz(image[0][:, :, :, 0], '/home/fengtong/models/official/projects/volumetric_models/visulization/img_channel_0_t5.nii.gz')
    # numpy_to_niigz(label[0][:, :, :, 1], '/home/fengtong/models/official/projects/volumetric_models/visulization/lb_class_1_t5.nii.gz')
    # numpy_to_niigz(label[0][:, :, :, 2], '/home/fengtong/models/official/projects/volumetric_models/visulization/lb_class_2_t5.nii.gz')



if __name__ == '__main__':
    # input_path = "/mnt/SSD2/feng/nnUNet_preprocessed/Task002_Heart/3d_tfrecord_data/fold*"
    # input_path = "/mnt/SSD2/feng/nnUNet_preprocessed/Task003_Liver/3d_tfrecord_data/fold*"
    # input_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task005_Prostate/3d_tfrecord_data/fold*"
    input_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task006_Lung/3d_tfrecord_data/fold*"
    # input_path = "/mnt/SSD2/feng/nnUNet_preprocessed/Task007_Pancreas/3d_tfrecord_data/fold*"
    
    read_tfrecord(input_path)
