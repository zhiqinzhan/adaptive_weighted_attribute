import caffe
import cv2
import time
import numpy as np

import common
import config
import evaluate

caffe.set_mode_gpu()
caffe.set_device(config.test_gpu_id)

net = caffe.Net(config.test_prototxt, config.test_model, caffe.TEST)

img_paths = common.dataset["test"]["img_paths"]
lab_array = common.dataset["test"]["lab_array"]

img_num = img_paths.shape[0]
lab_num = lab_array.shape[1]
lab_pred = np.zeros((img_num, lab_num), dtype=np.bool)
lab_gt = np.zeros((img_num, lab_num), dtype=np.bool)

for i in range(img_num):
    img = cv2.imread(img_paths[i])
    assert img is not None
    inputData = np.asarray([
        common.Pre_process.run(img),
        common.Pre_process.run(img, mirror=True)
    ])
    net.set_input_arrays(
        inputData.astype(np.float32),
        np.zeros([2, 1, 1, 1]).astype(np.float32)
    )
 
    start_time = time.time()
    out = net.forward()
    print "%10d: %f seconds" % (i, (time.time() - start_time))

    lab_pred[i] = np.mean(out['pred_attribute'], axis=0) > 0
    lab_gt[i] = lab_array[i] > 0

mA, acc_collect, challenging = evaluate.mA(lab_pred, lab_gt)
print "acc_collect", acc_collect
print "challenging", challenging
print "mA:", mA

accuracy, precision, recall, f1 = evaluate.example_based(lab_pred, lab_gt)
print "precision:", precision
print "recall:", recall
print "f1", f1
print "accuracy:", accuracy
