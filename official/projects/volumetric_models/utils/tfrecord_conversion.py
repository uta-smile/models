import os
import pickle

import numpy as np
import tensorflow as tf

# tfrecord feature keys
IMAGE_KEY = 'image/encoded'
CLASSIFICATION_LABEL_KEY = 'image/class/label'
IMAGE_SHAPE_KEY = 'image_shape'
LABEL_SHAPE_KEY = 'label_shape'


def convert_one_sample(data_folder, fold, file_name, save_path):
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
    # tfrecord_file = os.path.join(save_path, (tr_or_val + '_' + file_name + '.tfrecord'))
    tfrecord_file = os.path.join(save_path, ('fold{}'.format(fold) + '_val_' + file_name + '.tfrecord'))

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        writer.write(tf_example.SerializeToString())


def main(splits_file, fold, data_path, save_path):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    # tr_keys = splits[fold]['train']
    val_keys = splits[fold]['val']
    # tr_keys.sort()
    val_keys.sort()
    # for i, file_name in enumerate(tr_keys):
    #     print("converting training case:", i)
    #     convert_one_sample(data_path, file_name, save_path, tr_or_val='tr')
    for i, file_name in enumerate(val_keys):
        print("converting fold", fold, "validation case:", i)
        # convert_one_sample(data_path, file_name, save_path, tr_or_val='val')
        convert_one_sample(data_path, fold, file_name, save_path)
    print("done")


if __name__ == '__main__':
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task001_BrainTumour/"
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task002_Heart/"
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task003_Liver/"
    preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task004_Hippocampus/"
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task005_Prostate/"
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task006_Lung/"
    # preprocessed_task_path = "/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed/Task007_Pancreas/"

    network_architecture = '2d'  # 2d, 3d
    
    for i in range(5):
        fold = i  # 5-fold cross-validation. Fold: 0, 1, 2, 3, 4

        data_folder = preprocessed_task_path + "nnUNetData_plans_v2.1_2D_stage0"  # 2d: all tasks
        # data_folder = preprocessed_task_path + "nnUNetData_plans_v2.1_stage0"  # 3d: task 001, 002, 004, 005
        # data_folder = preprocessed_task_path + "nnUNetData_plans_v2.1_stage1"  # 3d: task 003, 006, 007
        
        splits_file = preprocessed_task_path + "splits_final.pkl"
        save_path = preprocessed_task_path + network_architecture + "_tfrecord_data"
        # save_path = "/mnt/SSD2/feng/nnUNet_preprocessed/Task002_Heart/" + network_architecture + "_tfrecord_data"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        main(splits_file, fold, data_folder, save_path)