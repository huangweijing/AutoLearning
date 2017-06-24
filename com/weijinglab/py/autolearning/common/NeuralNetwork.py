import numpy as np
import pickle

from com.weijinglab.py.autolearning.common import LearningFuncs


class NeuralNetwork:
    # X = np.zeros_like(3)
    # W1 = np.zeros_like(3, 6)
    # B1 = np.zeros_like(6)
    # W2 = np.zeros_like(6, 12)
    # B2 = np.zeros_like(12)
    # Y = np.zeros_like(8)
    # LearningFuncs.sigmoid(X)

    def __init__(self, input_size=3, output_size=8, hidden_layer_size=[3, 6], learning_rate=0.1):

        self.layer_info = []
        self.layer_info.append(input_size)
        self.layer_info.extend(hidden_layer_size)
        self.layer_info.append(output_size)

        print(self.layer_info)

        self.learning_rate = learning_rate

        self.W = []
        self.B = []

        for i in range(len(self.layer_info) - 1):
            self.W.append(np.random.randn(self.layer_info[i], self.layer_info[i + 1]))
            self.B.append(np.random.randn(self.layer_info[i + 1]))
            # self.W.append(np.zeros([self.layer_info[i], self.layer_info[i + 1]]))
            # self.B.append(np.zeros([self.layer_info[i + 1]]))

            #
            # self.W1 = np.random.randn(input_size, hidden_layer_size[0])
            # self.B1 = np.random.randn(hidden_layer_size[0])
            # self.W2 = np.random.randn(hidden_layer_size[0], hidden_layer_size[1])
            # self.B2 = np.random.randn(hidden_layer_size[1])
            # self.W3 = np.random.randn(hidden_layer_size[1], output_size)
            # self.B3 = np.random.randn(output_size)

        for i in range(len(self.W)):
            print("---W%s---Shape%s----" % (str(i), str(np.shape(self.W[i]))))
            print(self.W[i])
            print("---B%s---Shape%s----" % (str(i), str(np.shape(self.B[i]))))
            print(self.B[i])

    def predict(self, input_data):

        for i in range(len(self.B)):
            if i == 0:
                hidden_layer = np.dot(input_data, self.W[i]) + self.B[i]
            else:
                hidden_layer = np.dot(hidden_layer, self.W[i]) + self.B[i]

            if i != len(self.B) - 1:
                hidden_layer = LearningFuncs.sigmoid(hidden_layer)

        return LearningFuncs.softmax(hidden_layer)

        # L1 = np.dot(input_data, self.W1) + self.B1
        # L1 = LearningFuncs.sigmoid(L1)
        # L2 = np.dot(L1, self.W2) + self.B2
        # L2 = LearningFuncs.sigmoid(L2)
        # output_data = np.dot(L2, self.W3) + self.B3
        # # output_data = LearningFuncs.sigmoid(output_data)
        # output_data = LearningFuncs.softmax(output_data)
        #
        # return output_data

    def calc_loss(self, train_data_set, train_label_set):
        # for i in range(len(train_data_set)):
        output_data = self.predict(train_data_set)
        return LearningFuncs.cross_entropy_error(output_data, train_label_set)

    def update_gradient(self, train_data_set, train_label_set):
        func_grad = lambda m: self.calc_loss(train_data_set, train_label_set)

        grad_W = np.zeros_like(self.W)
        grad_B = np.zeros_like(self.B)

        for i in range(len(self.B)):
            grad_W[i] = LearningFuncs.gradient(func_grad, self.W[i])
            grad_B[i] = LearningFuncs.gradient(func_grad, self.B[i])

        self.W -= grad_W * self.learning_rate
        self.B -= grad_B * self.learning_rate

        # grad_W1 = LearningFuncs.gradient(func_grad, self.W1)
        # grad_W2 = LearningFuncs.gradient(func_grad, self.W2)
        # grad_W3 = LearningFuncs.gradient(func_grad, self.W3)
        # grad_B1 = LearningFuncs.gradient(func_grad, self.B1)
        # grad_B2 = LearningFuncs.gradient(func_grad, self.B2)
        # grad_B3 = LearningFuncs.gradient(func_grad, self.B3)
        #
        # self.W1 -= grad_W1 * lr
        # self.W2 -= grad_W2 * lr
        # self.W3 -= grad_W3 * lr
        # self.B1 -= grad_B1 * lr
        # self.B2 -= grad_B2 * lr
        # self.B3 -= grad_B3 * lr


def measure(neural_network: NeuralNetwork):
    good_answer_cnt = 0.0
    total_try_cnt = 1000.0

    for i in range(int(total_try_cnt)):
        r1 = np.random.random_integers(0, 1)
        r2 = np.random.random_integers(0, 1)
        r3 = np.random.random_integers(0, 1)
        bInt = r1 * 4 + r2 * 2 + r3
        bInt_arr = np.zeros(8)
        bInt_arr[bInt] = 1
        r = np.argmax(neural_network.predict([r1, r2, r3]))
        if r == bInt:
            good_answer_cnt = good_answer_cnt + 1

    return good_answer_cnt / total_try_cnt


def test():
    nn = NeuralNetwork(hidden_layer_size=[6, 8])
    result = nn.calc_loss([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    print(result)
    #
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])

    print("acc=" + str(measure(nn)))

    for i in range(1000000):
        v1 = np.random.random_integers(0, 1)
        v2 = np.random.random_integers(0, 1)
        v3 = np.random.random_integers(0, 1)
        n = v1 * 4 + v2 * 2 + v3
        n_arr = np.zeros(8)
        n_arr[n] = 1
        # print("training... %s%s%s, %s" % (v1, v2, v3, n))
        if i % 10000 == 0:
            acc = measure(nn)
            print("acc=" + str(acc))
            if acc >= 0.99:
                with open("nn_W", "wb") as w_f:
                    pickle.dump(nn.W, w_f, -1)

                with open("nn_B", "wb") as b_f:
                    pickle.dump(nn.B, b_f, -1)

                break

        nn.update_gradient([v1, v2, v3], n_arr)
    nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])

    # result = nn.calc_loss([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    # print(result)

    print("acc=" + str(measure(nn)))

    print(str(nn.predict([1, 1, 1])) + ":" + str(np.argmax(nn.predict([1, 1, 1]))))


    # v1 = np.random.random_integers(0, 1)
    # v2 = np.random.random_integers(0, 1)
    # v3 = np.random.random_integers(0, 1)
    # n = v1 * 4 + v2 * 2 + v3
    # n_arr = np.zeros(8)
    # n_arr[n] = 1
    # print([v1, v2, v3], n_arr)


def test2():
    nn = NeuralNetwork()
    with open("nn_W", "rb") as f_w:
        nn.W = pickle.load(f_w)

    with open("nn_B", "rb") as f_b:
        nn.B = pickle.load(f_b)

    print(np.argmax(nn.predict([1, 1, 0])))


test2()
