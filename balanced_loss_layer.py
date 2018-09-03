import numpy as np

import config
import python_loss_layer

attribute_size = config.pos_ratio.shape[0]
majority = config.pos_ratio > 0.5
majority_ratio = np.asarray([config.pos_ratio[i] if majority[i] else 1 - config.pos_ratio[i] \
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

class balancedSigmoidCrossEntropyLossLayer(
        python_loss_layer.SigmoidCrossEntropyLossLayer
    ):
    def backward(self, top, propagate_down, bottom):
        super.backward(top, propagate_down, bottom)
        dropGradient(bottom)

    

    

