import caffe
import time
import numpy as np

import common
import config
import evaluate

caffe.set_mode_gpu()
caffe.set_device(config.test_gpu_id)

net = caffe.Net(config.test_prototxt, config.test_model, caffe.TEST)

img_array = common.dataset["test"]["img_array"]
img_array_mirror = common.dataset["test_mirror"]["img_array"]
lab_array = common.dataset["test"]["lab_array"]

img_num = img_array.shape[0]
lab_num = lab_array.shape[1]
lab_pred = np.zeros((img_num, lab_num), dtype=np.bool)
lab_gt = np.zeros((img_num, lab_num), dtype=np.bool)

for i in range(img_num):
    inputData = np.asarray([img_array[i], img_array_mirror[i]])
    net.set_input_arrays(inputData.astype(np.float32), np.zeros([2, 1, 1, 1]).astype(np.float32))
 
    start_time = time.time()
    out = net.forward()
    print "%10d: %f seconds" % (i, (time.time() - start_time))

    lab_pred[i] = np.mean(out['pred_attribute'], axis=0) > 0
    lab_gt[i] = lab_array[i] > 0

accuracy, precision, recall, f1 = evaluate.example_based(lab_pred, lab_gt)
mA, acc_collect, challenging = evaluate.mA(lab_pred, lab_gt)

print "precision:", precision
print "recall:", recall
print "f1", f1
print "accuracy:", accuracy

print "mA:", mA
print "acc_collect", acc_collect
print "challenging", challenging
