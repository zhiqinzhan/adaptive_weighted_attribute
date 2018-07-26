img_width = 224
img_height = 224

batch_size = 18
batch_size_0 = batch_size / 2 # TODO

dataset_name = "PA-100K"

train_prototxt = 'model/PA-100K/joint_CE/solver.prototxt'
pretrained_model = 'model/resnet_50/ResNet-50-model.caffemodel'
train_gpu_id = 0

test_prototxt = 'model/PA-100K/joint_CE/deploy.prototxt'
test_model = 'output/PA-100K/joint_CE/saved_weighted_iter_130000.caffemodel'
test_gpu_id = 3

# TODO
# import numpy as np
# selected_attr = np.asarray(range(26), dtype=np.int)
# selected_attr = np.asarray([1, 7, 12, 19, 20, 21, 25], dtype=np.int)
