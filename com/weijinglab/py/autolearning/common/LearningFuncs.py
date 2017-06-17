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
    arr_minus_max = arr - np.max(arr)
    exp_sum = np.sum(np.exp(arr_minus_max))
    return np.exp(x - np.max(arr))/exp_sum


def cross_entropy_error(y, t):
    return np.sum(t * np.log(y) * -1)


print(cross_entropy_error([90 , 92444], [0, 1]))

