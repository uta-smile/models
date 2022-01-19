import os
import pickle

import numpy as np
import tensorflow as tf


def convert_one_sample(data_folder, file_name, save_path, tr_or_val):
    data = np.load(os.path.join(data_folder, (file_name + ".npz")))['data']
    image = data[:-1, :, :, :]
    label = data[-1, :, :, :]
    label = label.astype(np.int64)
    feature = {
        IMAGE_KEY: (tf.train.Feature(
            float_list=tf.train.FloatList(value=image.flatten()))),
        CLASSIFICATION_LABEL_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=label.flatten()))),
        IMAGE_SHAPE_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(image.shape)))),
        LABEL_SHAPE_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(label.shape))))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    tfrecord_file = os.path.join(save_path, (tr_or_val + '_' + file_name + '.tfrecord'))
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        writer.write(tf_example.SerializeToString())


def main(splits_file, fold, data_path, save_path):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    tr_keys = splits[fold]['train']
    val_keys = splits[fold]['val']
    tr_keys.sort()
    val_keys.sort()
    for file_name in tr_keys:
        convert_one_sample(data_path, file_name, save_path, tr_or_val='tr')
    for file_name in val_keys:
        convert_one_sample(data_path, file_name, save_path, tr_or_val='val')


if __name__ == '__main__':
    data_folder = "xxx/nnUNet_preprocessed/Task004_Hippocampus/nnUNetData_plans_v2.1_stage0"
    splits_file = "xxx/nnUNet_preprocessed/Task004_Hippocampus/splits_final.pkl"
    network_architecture = '3d'  # 2d, 3d
    fold = 0  # 5-fold cross-validation. Fold: 0, 1, 2, 3, 4

    tfrecord_folder_name = network_architecture + '_fold{}'.format(fold)
    save_path = "xxx/nnUNet_preprocessed/Task004_Hippocampus/" + tfrecord_folder_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # tfrecord feature keys
    IMAGE_KEY = 'image/encoded'
    CLASSIFICATION_LABEL_KEY = 'image/class/label'
    IMAGE_SHAPE_KEY = 'image_shape'
    LABEL_SHAPE_KEY = 'label_shape'

    main(splits_file, fold, data_folder, save_path)
