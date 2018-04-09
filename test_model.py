import numpy as np
import cv2
import caffe
import os
import time
import scipy.io as sio

caffe.set_mode_gpu()
caffe.set_device(0)

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
model_path = 'model/basic_saved_weighted__iter_130000.caffemodel'
net = caffe.Net(prototxt_path, model_path, caffe.TEST)

target_height = 224
target_width = 224
def pre_process(color_img, is_mirror=False):
    resized_img = cv2.resize(color_img, (target_width, target_height))
    if is_mirror:
        resized_img = cv2.flip(resized_img, 1)
    return np.transpose(resized_img, (2, 0, 1)) - mean

attri_num = 26
TP_sum = np.zeros(attri_num, dtype=np.int)
FN_sum = np.zeros(attri_num, dtype=np.int)
FP_sum = np.zeros(attri_num, dtype=np.int)
TN_sum = np.zeros(attri_num, dtype=np.int)

for i in xrange(img_num):
    img = cv2.imread(total_namelist[i])
    assert img is not None
    inputData = np.array([pre_process(img, False), pre_process(img, True)])
    net.set_input_arrays(inputData.astype(np.float32), np.zeros([2, 1, 1, 1]).astype(np.float32))
 
    start_time = time.time()
    out = net.forward()
    print "%s: %f seconds" % (total_namelist[i], (time.time() - start_time))

    score = np.mean(out['pred_attribute'], axis=0)
    pred = score > 0

    GT = attri_array[i] 
    TP = np.logical_and(pred, GT)
    FN = np.logical_and(np.logical_not(pred), GT)
    FP = np.logical_and(pred, np.logical_not(GT))
    TN = np.logical_and(np.logical_not(pred), np.logical_not(GT))

    TP_sum += TP
    FN_sum += FN
    FP_sum += FP
    TN_sum += TN

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

print "precision:", precision
print "mean precision:", np.mean(precision)
print "recall:", recall
print "mean recall:", np.mean(recall)
print "f1", f1
print "mean f1:", np.mean(f1)
print "accuracy:", accuracy
print "mean accuracy:", np.mean(accuracy)
