import caffe
import numpy as np


pos_ratio = np.array([
        45336, 1469, 92844, 5687, 34707, 30508, 34785, 4206, 18662,
        18115, 19301, 15926, 958, 56913, 43087, 5088, 14835, 10917,
        4219, 450, 1639, 3365, 71916, 16896, 11155, 595
    ], dtype=np.float32) / 100000

sigma = 0.5
# avg_weight = np.exp(0.5 / (sigma ** 2))
# pos_weight = np.exp((1.0 - pos_ratio) / (sigma ** 2)) / avg_weight
# neg_weight = np.exp(pos_ratio / (sigma ** 2)) / avg_weight
pos_weight = 0.5 / pos_ratio
neg_weight = 0.5 / (1.0 - pos_ratio)
assert(pos_weight.shape == neg_weight.shape)

def getLossWeight(label): # handle imbalance between positive and negative samples
    batch_size = label.shape[0]
    attribute_size = label.shape[1]
    assert(attribute_size == pos_weight.shape[0])

    w = np.zeros_like(label, dtype=np.float32)
    for b in range(batch_size):
        for a in range(attribute_size):
            w[b][a] = pos_weight[a] if label[b][a] > 0 else neg_weight[a]

    return w


gamma = 2.0

class SigmoidFocalLossLayer(caffe.Layer):
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

        top[0].data[...] = np.sum(-((1 - pt) ** gamma) * np.log(pt)) / pt.shape[0]
        self.pt = pt

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0]:
            y = bottom[1].data # ground-truth
            pt = self.pt
            bottom[0].diff[...] = y * ((1 - pt) ** gamma) * (gamma * pt * np.log(pt) + pt - 1) / pt.shape[0]


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

        top[0].data[...] = np.sum(-np.log(pt)) / pt.shape[0]
        self.pt = pt

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("Layer cannot backpropagate to label inputs.")
        if propagate_down[0]:
            y = bottom[1].data # ground-truth
            pt = self.pt
            bottom[0].diff[...] = y * (pt - 1) / pt.shape[0]


class EuclideanLossLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """
    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(self.diff**2) / bottom[0].num / 2.

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            bottom[i].diff[...] = sign * getLossWeight(bottom[1-i].data) * self.diff / bottom[i].num


class TrainValWeightedEuclideanLossLayer(caffe.Layer):
    """
    Compute the Wegithed Loss
    """
    count = 0
    batch = 0
    interval = 200
    val_loss = []
    pre_val_mean = 0
    norm_trend = None
    norm_loss = None

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.batch = bottom[0].num / 2  # Indicate first half as train, second half as val
        self.count = 0
        self.norm_trend = np.ones(bottom[0].data.shape[1])
        self.norm_loss = np.ones(bottom[0].data.shape[1])

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        self.val_loss.append(np.sum((getLossWeight(bottom[1].data[self.batch:]) * self.diff[self.batch:]) ** 2, axis=0)) # assert that bottom[1] is the ground truth
        top[0].data[...] = np.sum(self.diff[0:self.batch]**2) / self.batch / 2.
        self.count += 1

    def backward(self, top, propagate_down, bottom):
        for i in range(2):
            if not propagate_down[i]:
                continue
            if i == 0:
                sign = 1
            else:
                sign = -1
            if self.count == self.interval:
                self.pre_val_mean = np.mean(self.val_loss[0:self.interval], axis=0)
            if self.count >= 2 * self.interval and self.count % self.interval == 0:
                begin_index = self.count - self.interval
                end_index = self.count
                cur_val_mean = np.mean(self.val_loss[begin_index:end_index], axis=0)
                trend = abs(cur_val_mean - self.pre_val_mean) / cur_val_mean
                self.norm_trend = trend / np.mean(trend)
                self.norm_loss = cur_val_mean / np.mean(cur_val_mean)
                self.pre_val_mean = cur_val_mean
            weights = self.norm_trend * self.norm_loss
            norm_weights = weights / np.mean(weights)
            repmated_weight = np.tile(norm_weights, [self.batch, 1])
            bottom[i].diff[0:self.batch] = \
                sign * getLossWeight(bottom[1-i].data[0:self.batch]) * repmated_weight * self.diff[0:self.batch] / self.batch
