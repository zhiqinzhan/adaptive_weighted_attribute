import os
import cv2
import caffe
import numpy as np
import scipy.io as sio

import config

def load_resnet_mean(filepath="model/resnet_50"):
    st = open(os.path.join(filepath, 'ResNet_mean.binaryproto'), "rb").read()
    bp = caffe.io.caffe_pb2.BlobProto.FromString(st)
    return caffe.io.blobproto_to_array(bp)[0]

class Pre_process(object):

    width = config.img_width
    height = config.img_height
    mean = load_resnet_mean()
    if width != 224 or height != 224:
        mean = np.transpose(cv2.resize(np.transpose(mean, (1, 2, 0)), (width, height)), (2, 0, 1))

    @classmethod
    def run(cls, img, mirror=False):
        img = cv2.resize(img, (cls.width, cls.height))
        if mirror:
            img = cv2.flip(img, 1)
        return np.transpose(img, (2, 0, 1)) - cls.mean

def load_PA100K(datadir='data/PA-100K/'):

    def load_split(split_name):        
        img_paths = [os.path.join(datadir, 'release_data', name[0][0]) \
            for name in annotation[split_name+'_images_name']]
        
        lab_array = annotation[split_name+'_label'].astype(int)
        lab_array[lab_array == 0] = -1

        assert len(img_paths) == len(lab_array)
        return {'img_paths':img_paths, 'lab_array':lab_array}

    annotation = sio.loadmat(os.path.join(datadir, 'annotation.mat'))
    dataset = {}
    dataset['train'] = load_split('train')
    dataset['val'] = load_split('val')
    dataset['test'] = load_split('test')
    return dataset

def load_CelebA():
    raise Exception("Not implemented!")

def load_dataset(name):
    if name == "PA-100K":
        return load_PA100K()
    elif name == "CelebA":
        return load_CelebA()
    else:
        raise Exception("Unknown dataset: %s" % name)

dataset = load_dataset(config.dataset_name)

def load_caffe_batch(top, p0, split_name, idx_list):
    for p, idx in enumerate(idx_list):
        img = cv2.imread(dataset[split_name]["img_paths"][idx])
        if img is not None:
            top[0].data[p0+p] = Pre_process.run(img)
            top[1].data[p0+p] = dataset[split_name]["lab_array"][idx]
