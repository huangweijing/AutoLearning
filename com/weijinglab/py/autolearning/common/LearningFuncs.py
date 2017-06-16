# coding=utf-8
import numpy as np


# シグモイド関数
def sigmoid(x):
    arr = np.array(x)
    arr = 1 / (1 + np.exp(-1 * arr))
    return arr


# ソフトマックス関数
def softmax(x):
    arr = np.array(x)
    exp_sum = np.sum(np.exp(arr))
    return np.exp(x)/exp_sum


print(softmax([90 , 92]))


