import yaml
import os
import numpy as np

dataset_name = "CelebA"
model_name = "balanced_SE-ResNet"
train_gpu_id = 2
test_gpu_id = 3

with open(os.path.join("model", dataset_name, model_name, "config.yml"), "r") as f:
    yml_config = yaml.load(f)

img_width = yml_config["img_width"]
img_height = yml_config["img_height"]

img_mean_file = "model/resnet_50/ResNet_mean.binaryproto"

if "pre_process_method" in yml_config:
    pre_process_method = yml_config["pre_process_method"]
else:
    pre_process_method = "default"

batch_size = yml_config["batch_size"]
batch_size_0 = batch_size / 2 # TODO

train_prototxt = os.path.join("model", dataset_name, model_name, "solver.prototxt")
if "pretrained_model" in yml_config:
    pretrained_model = yml_config["pretrained_model"]
else:
    pretrained_model = 'model/resnet_50/ResNet-50-model.caffemodel'

test_prototxt = os.path.join("model", dataset_name, model_name, "deploy.prototxt")
test_model = os.path.join("output", dataset_name, model_name, "saved_weighted_iter_130000.caffemodel")

# TODO
# import numpy as np
# selected_attr = np.asarray(range(26), dtype=np.int)
# selected_attr = np.asarray([1, 7, 12, 19, 20, 21, 25], dtype=np.int)
