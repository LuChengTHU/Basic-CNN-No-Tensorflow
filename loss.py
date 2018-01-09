from __future__ import division
import numpy as np


class EuclideanLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        return 0.5 * np.mean(np.sum(np.square(input - target), axis=1))

    def backward(self, input, target):
        return (input - target) / len(input)


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        input_exp = np.exp(input)
        total = np.sum(input_exp, axis=1)
        total_list = np.repeat(total, input.shape[1], axis=0).reshape(input.shape)
        loss = np.multiply(np.log(input_exp / total_list), target)
        return -np.mean(np.sum(loss, axis=1))

    def backward(self, input, target):
        input_exp = np.exp(input)
        total = np.sum(input_exp, axis=1)
        total_list = np.repeat(total, input.shape[1], axis=0).reshape(input.shape)
        h = input_exp / total_list
        return -(target - h) / len(input)
