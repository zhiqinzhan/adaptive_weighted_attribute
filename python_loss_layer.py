import caffe
import numpy as np
import scipy


class SigmoidFocalLossLayer(caffe.Layer):
    gamma = 2.0
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        x = bottom[0].data
        y = bottom[1].data # ground-truth

        pt = scipy.special.expit(x)
        for i in range(pt.shape[0]):
            for j in range(pt.shape[1]):
                if y[i][j] < 0:
                    pt[i][j] = 1 - pt[i][j]
        
        self.pt = pt
        self.log_pt = x * ((y >= 0).astype(int) - (x >= 0)) - np.log(1 + np.exp(x - 2 * x * (x >= 0)))

        top[0].data[...] = np.sum(-((1 - pt) ** self.gamma) * self.log_pt) / pt.shape[0]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0]:
            y = bottom[1].data # ground-truth
            pt = self.pt
            bottom[0].diff[...] = y * ((1 - pt) ** self.gamma) * (self.gamma * pt * self.log_pt + pt - 1) / pt.shape[0]


class SigmoidCrossEntropyLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute crossentropy loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        x = bottom[0].data
        y = bottom[1].data # ground-truth

        pt = scipy.special.expit(x)
        for i in range(pt.shape[0]):
            for j in range(pt.shape[1]):
                if y[i][j] < 0:
                    pt[i][j] = 1 - pt[i][j]
        
        self.pt = pt
        self.log_pt = x * ((y >= 0).astype(int) - (x >= 0)) - np.log(1 + np.exp(x - 2 * x * (x >= 0)))

        top[0].data[...] = np.sum(-self.log_pt) / pt.shape[0]

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0]:
            y = bottom[1].data # ground-truth
            pt = self.pt
            bottom[0].diff[...] = y * (pt - 1) / pt.shape[0]


class TrainValWeightedSigmoidCrossEntropyLossLayer(caffe.Layer):
    interval = 200
    
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs (scores and labels) to compute crossentropy loss.")
        
        self.count = 0
        self.val_loss = []
        self.norm_weights = np.ones(bottom[0].data.shape[1])

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        x = bottom[0].data
        y = bottom[1].data # ground-truth

        pt = scipy.special.expit(x)
        for i in range(pt.shape[0]):
            for j in range(pt.shape[1]):
                if y[i][j] < 0:
                    pt[i][j] = 1 - pt[i][j]

        self.pt = pt
        self.log_pt = x * ((y >= 0).astype(int) - (x >= 0)) - np.log(1 + np.exp(x - 2 * x * (x >= 0)))

        batch_size = bottom[0].shape[0] / 2 # train: first half; val: second half;
        top[0].data[...] = np.sum(-self.log_pt[:batch_size]) / batch_size
        self.val_loss.append(np.sum(-self.log_pt[batch_size:], axis=0))
        self.count += 1

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0]:
            y = bottom[1].data # ground-truth
            pt = self.pt

            if self.count == self.interval:
                self.pre_val_mean = np.mean(self.val_loss[0:self.interval], axis=0)
            if self.count >= 2 * self.interval and self.count % self.interval == 0:
                cur_val_mean = np.mean(self.val_loss[self.count - self.interval : self.count], axis=0)
                trend = abs(cur_val_mean - self.pre_val_mean) / cur_val_mean
                norm_trend = trend / np.mean(trend)
                norm_loss = cur_val_mean / np.mean(cur_val_mean)
                weights = norm_trend * norm_loss
                self.norm_weights = weights / np.mean(weights)
                self.pre_val_mean = cur_val_mean
            batch_size = bottom[0].shape[0] / 2 # train: first half; val: second half;
            repmated_weight = np.tile(self.norm_weights, [batch_size, 1])
            bottom[0].diff[:batch_size] = \
                repmated_weight * y[:batch_size] * (pt[:batch_size] - 1) / batch_size
