from com.weijinglab.py.autolearning.common.NeuralNetwork import NeuralNetwork
import numpy as np


# nn = NeuralNetwork()

def measure(neural_network: NeuralNetwork):
    good_answer_cnt = 0.0
    total_try_cnt = 1000.0

    for i in range(total_try_cnt):
        v1 = np.random.random_integers(0, 1)
        v2 = np.random.random_integers(0, 1)
        v3 = np.random.random_integers(0, 1)
        n = v1 * 4 + v2 * 2 + v3
        n_arr = np.zeros(8)
        n_arr[n] = 1
        result = np.argmax(neural_network.predict(n_arr))
        if result == n:
            good_answer_cnt = good_answer_cnt + 1

    return good_answer_cnt / total_try_cnt
