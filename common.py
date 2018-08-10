import os
import cv2
import caffe
import numpy as np
import scipy.io as sio

import config

def load_img_mean():
    st = open(config.img_mean_file, "rb").read()
    bp = caffe.io.caffe_pb2.BlobProto.FromString(st)
    return caffe.io.blobproto_to_array(bp)[0]

class Pre_process(object):

    width = config.img_width
    height = config.img_height
    mean = load_img_mean()
    if width != 224 or height != 224:
        mean = np.transpose(cv2.resize(np.transpose(mean, (1, 2, 0)), (width, height)), (2, 0, 1))

    @classmethod
    def run(cls, img, mirror=False):
        if config.pre_process_method == "default":
            return cls.default(img, mirror)
        elif config.pre_process_method == "keep_ratio":
            return cls.keep_ratio(img, mirror)
        else:
            raise Exception("Not implemented!")

    @classmethod
    def default(cls, img, mirror=False):
        img = cv2.resize(img, (cls.width, cls.height))
        if mirror:
            img = cv2.flip(img, 1)
        return np.transpose(img, (2, 0, 1)) - cls.mean
    
    @classmethod
    def keep_ratio(cls, img, mirror=False):
        h, w = img.shape[0 : 2]
        if int(w * cls.height / h) > cls.width:
            h = int(h * cls.width / w)
            w = cls.width
            h0 = int((cls.height - h) / 2)
            w0 = 0
        else:
            w = int(w * cls.height / h)
            h = cls.height
            h0 = 0
            w0 = int((cls.width - w) / 2)
        img = cv2.resize(img, (w, h))
        if mirror:
            img = cv2.flip(img, 1)
        img = np.transpose(img, (2, 0, 1)) - cls.mean[:, h0:h0+h, w0:w0+w]
        result = np.zeros((3, cls.height, cls.width), dtype=np.float32)
        result[:, h0:h0+h, w0:w0+w] = img
        return result

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
