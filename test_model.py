import numpy as np
import cv2
import caffe
import os
import time
import scipy.io as sio

selected_attr = np.asarray(range(26), dtype=np.int)
# selected_attr = np.asarray([1, 7, 12, 19, 20, 21, 25], dtype=np.int)

caffe.set_mode_gpu()
caffe.set_device(3)

datadir = 'data/PA-100K/'
annotation = sio.loadmat(os.path.join(datadir, 'annotation.mat'))

imgset = 'test'
total_namelist = [os.path.join(datadir, 'release_data', name[0][0]) \
                  for name in annotation[imgset+'_images_name']]
img_num = len(total_namelist)

attri_array = annotation[imgset+'_label']
attri_array = attri_array.astype(int)

mean_file = 'model/resnet_50/ResNet_mean.binaryproto'
proto_data = open(mean_file, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]

prototxt_path = 'deploy_resnet.prototxt'
model_path = 'model/CrossEntropy_saved_weighted__iter_130000.caffemodel'
net = caffe.Net(prototxt_path, model_path, caffe.TEST)

target_height = 448
target_width = 224
resized_mean = np.transpose(cv2.resize(np.transpose(mean, (1, 2, 0)), (target_width, target_height)), (2, 0, 1))
def pre_process(color_img, is_mirror=False):
    resized_img = cv2.resize(color_img, (target_width, target_height))
    if is_mirror:
        resized_img = cv2.flip(resized_img, 1)
    # resized_img = (resized_img > np.random.randint(256, size=resized_img.shape)) * 255
    return np.transpose(resized_img, (2, 0, 1)) - resized_mean

attri_num = len(selected_attr)
attr_pred = np.zeros((img_num, attri_num), dtype=np.bool)
attr_gt = np.zeros((img_num, attri_num), dtype=np.bool)
"""
TP_sum = np.zeros(attri_num, dtype=np.int)
FN_sum = np.zeros(attri_num, dtype=np.int)
FP_sum = np.zeros(attri_num, dtype=np.int)
TN_sum = np.zeros(attri_num, dtype=np.int)
"""

for i in xrange(img_num):
    img = cv2.imread(total_namelist[i])
    assert img is not None
    
    # batch = 18
    batch = 2
    inputData = np.array([pre_process(img, j % 2 == 1) for j in range(batch)])
    net.set_input_arrays(inputData.astype(np.float32), np.zeros([batch, 1, 1, 1]).astype(np.float32))
 
    start_time = time.time()
    out = net.forward()
    print "%s: %f seconds" % (total_namelist[i], (time.time() - start_time))

    pred = np.mean(out['pred_attribute'], axis=0) > 0
    GT = np.asarray([attri_array[i][j] for j in selected_attr], dtype=np.int)

    attr_pred[i] = pred
    attr_gt[i] = GT

    # print GT
    # print out['pred_attribute']
    
    """
    TP = np.logical_and(pred, GT)
    FN = np.logical_and(np.logical_not(pred), GT)
    FP = np.logical_and(pred, np.logical_not(GT))
    TN = np.logical_and(np.logical_not(pred), np.logical_not(GT))

    TP_sum += TP
    FN_sum += FN
    FP_sum += FP
    TN_sum += TN
    """

from evaluate import example_based, mA
accuracy, precision, recall, f1 = example_based(attr_pred, attr_gt)
mA, acc_collect, challenging = mA(attr_pred, attr_gt)

"""
print "TP:", TP_sum
print "FN:", FN_sum
print "FP:", FP_sum
print "TN:", TN_sum

TP_sum = TP_sum.astype(np.float)
FN_sum = FN_sum.astype(np.float)
FP_sum = FP_sum.astype(np.float)
TN_sum = TN_sum.astype(np.float)

precision = TP_sum / (TP_sum + FP_sum)
recall = TP_sum / (TP_sum + FN_sum)
f1 = (2 * TP_sum) / (2 * TP_sum + FP_sum + FN_sum)
accuracy = (TP_sum + TN_sum) / img_num
"""

print "precision:", precision
# print "mean precision:", np.mean(precision)
print "recall:", recall
# print "mean recall:", np.mean(recall)
print "f1", f1
# print "mean f1:", np.mean(f1)
print "accuracy:", accuracy
# print "mean accuracy:", np.mean(accuracy)

print "mA:", mA
print "acc_collect", acc_collect
print "challenging", challenging
