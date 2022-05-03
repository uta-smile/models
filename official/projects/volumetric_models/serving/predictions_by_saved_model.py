# /home/feng/Desktop/tf_unet/ckpt-5.data-00000-of-00001
# /home/feng/Desktop/tf_unet/ckpt-5.index
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf



def inference_apply_nonlin(torch_input):
    export_dir_path = '/home/fengtong/models/official/projects/volumetric_models/serving/exported_model_1/saved_model'

    b, c, x, y, z = torch_input.shape
    torch_input = torch_input.view(b, x, y, z, c)
    np_input = torch_input.numpy()
    tf_input = tf.convert_to_tensor(np_input)
  
    imported = tf.saved_model.load(export_dir_path)
    model_fn = imported.signatures['serving_default']
    tf_output = model_fn(inputs=tf_input)
    
    np_output = tf_output['logits'].numpy()
    torch_output = torch.from_numpy(np_output)
    b, x, y, z, c = torch_output.shape
    torch_output = torch_output.view(b, c, x, y, z)
    return torch_output


if __name__ == '__main__':
    # pred = self.inference_apply_nonlin(self(x))
    # x: <class 'torch.Tensor'>, torch.Size([1, 1, 40, 56, 40])
    # pred: <class 'torch.Tensor'>, torch.Size([1, 3, 40, 56, 40])

    import torch
    import tensorflow as tf

    x = torch.zeros(1, 1, 40, 56, 40)
    pred = inference_apply_nonlin(x)
    print(x, type(x), x.shape)
    print(pred, type(pred), pred.shape)
    # <class 'torch.Tensor'> torch.Size([1, 1, 40, 56, 40])
    # <class 'torch.Tensor'> torch.Size([1, 3, 40, 56, 40])



    # export_dir_path = '/home/feng/Desktop/models-master/official/vision/beta/projects/volumetric_models/serving/exported_model1/saved_model'
    # # input_type = XX
    # input_images = tf.random.uniform(shape=[1, 40, 56, 40, 1])
    # print("################", input_images.dtype)
    # # input_images = tf.cast(input_images, dtype=tf.uint8)
    # imported = tf.saved_model.load(export_dir_path)
    # model_fn = imported.signatures['serving_default']
    # output = model_fn(inputs=input_images)

    # print(type(input_images), input_images.shape)
    # # print(input_images)
    # print(type(output), len(output))
    # print(type(output['logits']), output['logits'].shape)


    # pytorch_tensor = torch.zeros(10)
    # np_tensor = pytorch_tensor.numpy()
    # tf_tensor = tf.convert_to_tensor(np_tensor)
    # np_tensor = pytorch_tensor.numpy()
    # pytorch_tensor = torch.from_numpy(np_tensor)
    # print(type(pytorch_tensor), pytorch_tensor.shape)
    # print(type(np_tensor), np_tensor.shape)
    # print(type(tf_tensor), tf_tensor.shape)
    # print(type(np_tensor), np_tensor.shape)
    # print(type(pytorch_tensor), pytorch_tensor.shape)
