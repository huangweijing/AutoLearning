import numpy as np
import pickle

from com.weijinglab.py.autolearning.common import LearningFuncs


class NeuralNetwork:
    def __init__(self, input_size=3, output_size=8, hidden_layer_size=[3, 6], learning_rate=0.1):

        self.layer_info = []
        self.layer_info.append(input_size)
        self.layer_info.extend(hidden_layer_size)
        self.layer_info.append(output_size)

        self.learning_rate = learning_rate

        self.W = []
        self.B = []

        for i in range(len(self.layer_info) - 1):
            self.W.append(np.random.randn(self.layer_info[i], self.layer_info[i + 1]))
            self.B.append(np.random.randn(self.layer_info[i + 1]))

    def predict(self, input_data):

        for i in range(len(self.B)):
            if i == 0:
                hidden_layer = np.dot(input_data, self.W[i]) + self.B[i]
            else:
                hidden_layer = np.dot(hidden_layer, self.W[i]) + self.B[i]

            if i != len(self.B) - 1:
                hidden_layer = LearningFuncs.sigmoid(hidden_layer)

        return LearningFuncs.softmax(hidden_layer)

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


def generate_input_data(i):
    r1 = np.random.random_integers(0, 1)
    r2 = np.random.random_integers(0, 1)
    r3 = np.random.random_integers(0, 1)
    return [r1, r2, r3]


def validate_output_data(input_data, output_data):
    correct_result = input_data[0] * 4 + input_data[1] * 2 + input_data[2]
    predicted_result = np.argmax(output_data)
    return correct_result == predicted_result


def measure(neural_network: NeuralNetwork
            , generate_input, validate_output):
    good_answer_cnt = 0.0
    total_try_cnt = 1000.0

    for i in range(int(total_try_cnt)):

        input_data = generate_input(i)
        result = neural_network.predict(input_data)

        if validate_output(input_data, result):
            good_answer_cnt = good_answer_cnt + 1

    return good_answer_cnt / total_try_cnt


# def measure(neural_network: NeuralNetwork):
#     good_answer_cnt = 0.0
#     total_try_cnt = 1000.0
#
#     for i in range(int(total_try_cnt)):
#         r1 = np.random.random_integers(0, 1)
#         r2 = np.random.random_integers(0, 1)
#         r3 = np.random.random_integers(0, 1)
#         bInt = r1 * 4 + r2 * 2 + r3
#         bInt_arr = np.zeros(8)
#         bInt_arr[bInt] = 1
#         r = np.argmax(neural_network.predict([r1, r2, r3]))
#         if r == bInt:
#             good_answer_cnt = good_answer_cnt + 1
#
#     return good_answer_cnt / total_try_cnt


def test():
    nn = NeuralNetwork(hidden_layer_size=[6, 8])
    result = nn.calc_loss([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
    print(result)

    for i in range(1000000):
        v1 = np.random.random_integers(0, 1)
        v2 = np.random.random_integers(0, 1)
        v3 = np.random.random_integers(0, 1)
        n = v1 * 4 + v2 * 2 + v3
        n_arr = np.zeros(8)
        n_arr[n] = 1
        # print("training... %s%s%s, %s" % (v1, v2, v3, n))
        if i % 10000 == 0:
            acc = measure(nn, generate_input_data, validate_output_data)
            print("acc=" + str(acc))
            if acc >= 0.99:
                # 最適化した係数を保存する
                # with open("nn_W", "wb") as w_f:
                #     pickle.dump(nn.W, w_f, -1)
                #
                # with open("nn_B", "wb") as b_f:
                #     pickle.dump(nn.B, b_f, -1)

                break

        nn.update_gradient([v1, v2, v3], n_arr)
    nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])

    print("acc=" + str(measure(nn, generate_input_data, validate_output_data)))

    print(str(nn.predict([1, 1, 1])) + ":" + str(np.argmax(nn.predict([1, 1, 1]))))

# # 最適化した係数をファイルから読み出す
# def test2():
#     nn = NeuralNetwork()
#     with open("nn_W", "rb") as f_w:
#         nn.W = pickle.load(f_w)
#
#     with open("nn_B", "rb") as f_b:
#         nn.B = pickle.load(f_b)
#
#     print(np.argmax(nn.predict([1, 1, 0])))
#
#
# test2()


test()
