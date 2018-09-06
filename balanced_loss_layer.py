import numpy as np

import config
import python_loss_layer

pos_ratio = common.dataset["train"]["pos_ratio"]
attribute_size = pos_ratio.shape[0]
majority = pos_ratio > 0.5
majority_ratio = np.asarray([pos_ratio[i] if majority[i] else 1 - pos_ratio[i] \
    for i in range(attribute_size)], dtype=np.float32)
majority_drop_rate = 2 - 1 / majority_ratio

def dropGradient(bottom):
    batch_size = bottom[0].shape[0]
    assert(attribute_size == bottom[0].shape[1])
    y = bottom[1].data > 0 # ground-truth
    r = np.random.rand(batch_size, attribute_size)
    for b in range(batch_size):
        for a in range(attribute_size):
            if y[b][a] == majority[a] and r[b][a] < majority_drop_rate[a]:
                bottom[0].diff[b][a] = 0

def dropGradient_HM(bottom, threshold, loss, batch_size=None):
    if batch_size is None:
        batch_size = bottom[0].shape[0]
    assert(attribute_size == bottom[0].shape[1])
    y = bottom[1].data > 0 # ground-truth
    r = np.random.rand(batch_size, attribute_size)
    for b in range(batch_size):
        for a in range(attribute_size):
            if y[b][a] == majority[a]:
                if loss[b][a] < threshold[a]:
                    bottom[0].diff[b][a] = 0
                    if r[b][a] > majority_drop_rate[a]:
                        threshold[a] = loss[b][a]
                else:
                    if r[b][a] < majority_drop_rate[a]:
                        threshold[a] = loss[b][a]

class balancedSigmoidCrossEntropyLossLayer(
        python_loss_layer.SigmoidCrossEntropyLossLayer
    ):
    def backward(self, top, propagate_down, bottom):
        super(balancedSigmoidCrossEntropyLossLayer, self).backward(top, propagate_down, bottom)
        dropGradient(bottom)

class balancedTrainValWeightedSigmoidCrossEntropyLossLayer(
        python_loss_layer.TrainValWeightedSigmoidCrossEntropyLossLayer
    ):
    def backward(self, top, propagate_down, bottom):
        super(balancedTrainValWeightedSigmoidCrossEntropyLossLayer, self).backward(top, propagate_down, bottom)
        dropGradient(bottom)

class HM_SigmoidCrossEntropyLossLayer(
        python_loss_layer.SigmoidCrossEntropyLossLayer
    ):
    def setup(self, bottom, top):
        super(HM_SigmoidCrossEntropyLossLayer, self).setup(bottom, top)
        self.threshold = np.zeros(attribute_size, dtype=np.float32)
    def backward(self, top, propagate_down, bottom):
        super(HM_SigmoidCrossEntropyLossLayer, self).backward(top, propagate_down, bottom)
        dropGradient_HM(bottom, self.threshold, -self.log_pt)

class HM_TrainValWeightedSigmoidCrossEntropyLossLayer(
        python_loss_layer.TrainValWeightedSigmoidCrossEntropyLossLayer
    ):
    def setup(self, bottom, top):
        super(HM_TrainValWeightedSigmoidCrossEntropyLossLayer, self).setup(bottom, top)
        self.threshold = np.zeros(attribute_size, dtype=np.float32)
    def backward(self, top, propagate_down, bottom):
        super(HM_TrainValWeightedSigmoidCrossEntropyLossLayer, self).backward(top, propagate_down, bottom)
        dropGradient_HM(bottom, self.threshold, -self.log_pt, batch_size=bottom[0].shape[0]/2)
