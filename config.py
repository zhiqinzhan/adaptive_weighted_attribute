import yaml
import os
import numpy as np

dataset_name = "PA-100K"
model_name = "SE-ResNet_attentionV2"
train_gpu_id = 0
test_gpu_id = 3

pos_ratio = np.array([
        45336, 1469, 92844, 5687, 34707, 30508, 34785, 4206, 18662,
        18115, 19301, 15926, 958, 56913, 43087, 5088, 14835, 10917,
        4219, 450, 1639, 3365, 71916, 16896, 11155, 595
    ], dtype=np.float32) / 100000

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
