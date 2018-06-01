import numpy as np
import sys
import cv2
import caffe
import os
import random
import time
import matplotlib.pyplot as plt
import scipy.io as sio

datadir = 'data/PA-100K/'
annotation = sio.loadmat(os.path.join(datadir, 'annotation.mat'))

mean_file = 'model/resnet_50/ResNet_mean.binaryproto'
proto_data = open(mean_file, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean = caffe.io.blobproto_to_array(a)[0]

selected_attr = np.asarray(range(26), dtype=np.int)
# selected_attr = np.asarray([1, 7, 12, 19, 20, 21, 25], dtype=np.int)

resized_mean = np.transpose(cv2.resize(np.transpose(mean, (1, 2, 0)), (224, 448)), (2, 0, 1))
def pre_process(color_img):
    resized_img = cv2.resize(color_img, (224, 448))
    # resized_img = (resized_img > np.random.randint(256, size=resized_img.shape)) * 255
    return np.transpose(resized_img, (2, 0, 1)) - resized_mean


def showImage(img, points=None, bbox=None):
    if points is not None:
        for i in range(0, points.shape[0]/2):
            cv2.circle(img, (int(round(points[i*2])), int(points[i*2+1])), 1, (0,0,255), 2)
    if bbox is not None:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[2])), (int(bbox[1]), int(bbox[3])), (0,0,255), 2)
    plt.figure()
    plt.imshow(img)
    plt.show()
    print 'here'


class ValLayer(caffe.Layer):

    attri_num = 26

    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    total_namelist = []
    attri_array =[]
    # landmark_array =[]
    img_num = 0

    target_height = 448
    target_witdh = 224
    # mean = []
    batch = 8
    imgset = 'val'


    def setup(self, bottom, top):
        self.total_namelist = [os.path.join(datadir, 'release_data', name[0][0]) \
                               for name in annotation[self.imgset+'_images_name']]
        self.img_num = len(self.total_namelist)
        
        self.attri_array = annotation[self.imgset+'_label']
        self.attri_array = self.attri_array.astype(int)
        self.attri_array[self.attri_array == 0] = -1

        top[0].reshape(self.batch, 3, self.target_height, self.target_witdh)
        top[1].reshape(self.batch, len(selected_attr))

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in range(self.batch):
            # print i
            idx = random.randint(0, self.img_num-1)
            im = cv2.imread(self.total_namelist[idx])
            if im is not None:
                top[0].data[i] = pre_process(im)
                for k in range(0, len(selected_attr)):
                    top[1].data[i][k] = self.attri_array[idx][int(selected_attr[k])]

    def backward(self, top, propagate_down, bottom):
        pass


class TrainLayer(ValLayer):
    imgset = 'train'

"""
class TrainLayer(caffe.Layer):

    attri_num = 26
    target_height = 224
    target_witdh = 224
    batch = 18
    imgset = 'train'

    def setup(self, bottom, top):
        # total_namelist
        # attri_array
        # img_num
        # pos_sample_list

        self.total_namelist = [os.path.join(datadir, 'release_data', name[0][0]) \
                               for name in annotation[self.imgset+'_images_name']]
        self.img_num = len(self.total_namelist)
        
        self.attri_array = annotation[self.imgset+'_label']
        self.attri_array = self.attri_array.astype(int)
        self.attri_array[self.attri_array == 0] = -1

        self.pos_sample_list = [ [] for i in range(self.attri_num) ]
        for i in range(self.img_num):
            for j in range(self.attri_num):
                if self.attri_array[i][j] > 0:
                    self.pos_sample_list[j].append(i)
        self.pos_sample_list = [ np.asarray(x) for x in self.pos_sample_list ]

        top[0].reshape(self.batch, 3, self.target_height, self.target_witdh)
        top[1].reshape(self.batch, len(selected_attr))

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        sample_idx = []

        # for i in range(self.attri_num):
        #     sample_idx.append(np.random.choice(self.pos_sample_list[i]))

        while len(sample_idx) < self.batch:
            for i in range(len(selected_attr)):
                sample_idx.append(np.random.choice(self.pos_sample_list[selected_attr[i]]))

        random.shuffle(sample_idx)
        sample_idx = sample_idx[:self.batch]

        for i, idx in enumerate(sample_idx):
            im = cv2.imread(self.total_namelist[idx])
            if im is not None:
                top[0].data[i] = pre_process(im)
                for k in range(len(selected_attr)):
                    top[1].data[i][k] = self.attri_array[idx][selected_attr[k]]

    def backward(self, top, propagate_down, bottom):
        pass
"""


class JointAttributeLayer(caffe.Layer):

    attri_num = 26
    point_num = 5

    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    train_total_namelist = []
    train_attri_array = []
    train_img_num = 0

    val_total_namelist = []
    val_attri_array =[]
    val_img_num = 0
    target_height = 448
    target_witdh = 224
    batch = 4
    train_imgset = 'train'
    val_imgset = 'val'

    def do_setup(self, imgset, attri_num):
        total_namelist = [os.path.join(datadir, 'release_data', name[0][0]) \
                          for name in annotation[imgset+'_images_name']]
        img_num = len(total_namelist)
        
        attri_array = annotation[imgset+'_label']
        attri_array = attri_array.astype(int)
        attri_array[attri_array == 0] = -1

        return total_namelist, img_num, attri_array

    def setup(self, bottom, top):
        self.train_total_namelist, self.train_img_num, self.train_attri_array = \
            self.do_setup(self.train_imgset, self.attri_num)
        self.val_total_namelist, self.val_img_num, self.val_attri_array = \
            self.do_setup(self.val_imgset, self.attri_num)
        top[0].reshape(2 * self.batch, 3, self.target_height, self.target_witdh)
        top[1].reshape(2 * self.batch, len(selected_attr))

    def reshape(self, bottom, top):
        pass

    def do_forward(self, top, begin_index, total_namelist, img_num, attri_array):
        for i in range(self.batch):
            # print i
            idx = random.randint(0, img_num-1)
            im = cv2.imread(total_namelist[idx])
            if im is not None:
                top[0].data[i + begin_index] = pre_process(im)
                for k in range(0, len(selected_attr)):
                    top[1].data[i + begin_index][k] = attri_array[idx][int(selected_attr[k])]

    def forward(self, bottom, top):
         self.do_forward(top, 0, self.train_total_namelist, self.train_img_num,
                         self.train_attri_array)
         self.do_forward(top, self.batch, self.val_total_namelist, self.val_img_num,
                         self.val_attri_array)

    def backward(self, top, propagate_down, bottom):
        pass

