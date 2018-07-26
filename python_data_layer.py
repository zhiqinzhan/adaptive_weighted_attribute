import caffe
import numpy as np

import config
import common

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        split_name = params["split_name"]

        self.img_array = common.dataset[split_name]["img_array"]
        self.lab_array = common.dataset[split_name]["lab_array"]
        
        self.batch_size = config.batch_size

        top[0].reshape(self.batch_size, 3, config.img_height, config.img_width)
        top[1].reshape(self.batch_size, self.lab_array.shape[1])

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            idx = np.random.randint(0, self.img_array.shape[0])
            top[0].data[i] = self.img_array[idx]
            top[1].data[i] = self.lab_array[idx]

    def backward(self, top, propagate_down, bottom):
        pass

class JointDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params = eval(self.param_str)
        split_name0 = params["split_name0"]
        split_name1 = params["split_name1"]

        self.img_array0 = common.dataset[split_name0]["img_array"]
        self.lab_array0 = common.dataset[split_name0]["lab_array"]

        self.img_array1 = common.dataset[split_name1]["img_array"]
        self.lab_array1 = common.dataset[split_name1]["lab_array"]
        
        self.batch_size = config.batch_size
        self.batch_size0 = config.batch_size_0

        top[0].reshape(self.batch_size, 3, config.img_height, config.img_width)
        top[1].reshape(self.batch_size, self.lab_array0.shape[1])

    def reshape(self, bottom, top):
        pass

    def do_forward(self, top, begin, end, img_array, lab_array):
        for i in range(begin, end):
            idx = np.random.randint(0, img_array.shape[0])
            top[0].data[i] = img_array[idx]
            top[1].data[i] = lab_array[idx]
    
    def forward(self, bottom, top):
        self.do_forward(top, 0, self.batch_size0, self.img_array0, self.lab_array0)
        self.do_forward(top, self.batch_size0, self.batch_size, self.img_array1, self.lab_array1)

    def backward(self, top, propagate_down, bottom):
        pass
