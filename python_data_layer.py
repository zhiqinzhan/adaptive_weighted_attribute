import caffe
import numpy as np

import config
import common

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.split_name = params["split_name"]
        self.batch_size = config.batch_size
        self.img_num = common.dataset[self.split_name]["img_array"].shape[0]
        self.lab_num = common.dataset[self.split_name]["lab_array"].shape[1]

        top[0].reshape(self.batch_size, 3, config.img_height, config.img_width)
        top[1].reshape(self.batch_size, self.lab_num)

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        idx_list = np.random.randint(0, self.img_num, size=self.batch_size)
        common.load_caffe_batch(top, 0, self.split_name, idx_list)

    def backward(self, top, propagate_down, bottom):
        pass

class JointDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.split_name_0 = params["split_name_0"]
        self.split_name_1 = params["split_name_1"]
        self.batch_size = config.batch_size
        self.batch_size_0 = config.batch_size_0
        self.img_num_0 = common.dataset[self.split_name_0]["img_array"].shape[0]
        self.img_num_1 = common.dataset[self.split_name_1]["img_array"].shape[0]
        self.lab_num = common.dataset[self.split_name_0]["lab_array"].shape[1]

        top[0].reshape(self.batch_size, 3, config.img_height, config.img_width)
        top[1].reshape(self.batch_size, self.lab_num)

    def reshape(self, bottom, top):
        pass
    
    def forward(self, bottom, top):
        idx_list_0 = np.random.randint(0, self.img_num_0, size=self.batch_size_0)
        common.load_caffe_batch(top, 0, self.split_name_0, idx_list_0)
        idx_list_1 = np.random.randint(0, self.img_num_1, size=self.batch_size_1)
        common.load_caffe_batch(top, self.batch_size_0, self.split_name_1, idx_list_1)

    def backward(self, top, propagate_down, bottom):
        pass
