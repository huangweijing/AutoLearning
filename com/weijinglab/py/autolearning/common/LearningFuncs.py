# coding=utf-8
import numpy as np


# シグモイド関数
def sigmoid(x):
    SIGMOID_MAX = 50
    arr = np.array(x)

    for n in np.nditer(arr, op_flags=['readwrite']):
        if n[...] >= SIGMOID_MAX:
            n[...] = SIGMOID_MAX
        elif n[...] <= -SIGMOID_MAX:
            n[...] = -SIGMOID_MAX

    arr = 1 / (1 + np.exp(-1 * arr))

    return arr


# ソフトマックス関数
def softmax(x):
    arr = np.array(x)
    arr_minus_max = arr - np.max(arr)
    exp_sum = np.sum(np.exp(arr_minus_max))
    return np.exp(x - np.max(arr)) / exp_sum


def cross_entropy_error(y, t):
    sum = 0
    for i in range(len(t)):
        sum += t[i] * np.log(y[i]) * -1
    return sum
    # return np.sum(t * np.log(y) * -1)


# 関数funcがxにおいて、各次元のdxを求める
def gradient(func, x):
    h = 0.0001
    result = np.zeros_like(x)

    for idx in range(len(x)):
        xn = x[idx]
        x[idx] = xn + h
        func_y1 = func(x)

        x[idx] = xn - h
        func_y2 = func(x)

        x[idx] = xn

        d = (func_y1 - func_y2) / (h * 2)
        result[idx] = d
    return result

