import numpy as np

from com.weijinglab.py.autolearning.common import LearningFuncs


class NeuralNetwork:
    # X = np.zeros_like(3)
    # W1 = np.zeros_like(3, 6)
    # B1 = np.zeros_like(6)
    # W2 = np.zeros_like(6, 12)
    # B2 = np.zeros_like(12)
    # Y = np.zeros_like(8)
    # LearningFuncs.sigmoid(X)

    def __init__(self, input_size=3, output_size=8, hidden_layer_size=[6, 16]):
        self.W1 = np.random.randn(input_size, hidden_layer_size[0])
        self.B1 = np.random.randn(hidden_layer_size[0])
        self.W2 = np.random.randn(hidden_layer_size[0], hidden_layer_size[1])
        self.B2 = np.random.randn(hidden_layer_size[1])
        self.W3 = np.random.randn(hidden_layer_size[1], output_size)
        self.B3 = np.random.randn(output_size)

    def predict(self, input_data):
        L1 = np.dot(input_data, self.W1) + self.B1
        L1 = LearningFuncs.sigmoid(L1)
        L2 = np.dot(L1, self.W2) + self.B2
        L2 = LearningFuncs.sigmoid(L2)
        output_data = np.dot(L2, self.W3) + self.B3
        # output_data = LearningFuncs.sigmoid(output_data)
        output_data = LearningFuncs.softmax(output_data)

        return output_data

    def calc_loss(self, train_data_set, train_label_set):
        # for i in range(len(train_data_set)):
        output_data = self.predict(train_data_set)
        return LearningFuncs.cross_entropy_error(output_data, train_label_set)

    def update_gradient(self, train_data_set, train_label_set):
        func_grad = lambda M: self.calc_loss(train_data_set, train_label_set)
        lr = 0.5

        grad_W1 = LearningFuncs.gradient(func_grad, self.W1)
        grad_W2 = LearningFuncs.gradient(func_grad, self.W2)
        grad_W3 = LearningFuncs.gradient(func_grad, self.W3)
        grad_B1 = LearningFuncs.gradient(func_grad, self.B1)
        grad_B2 = LearningFuncs.gradient(func_grad, self.B2)
        grad_B3 = LearningFuncs.gradient(func_grad, self.B3)

        self.W1 -= grad_W1 * lr
        self.W2 -= grad_W2 * lr
        self.W3 -= grad_W3 * lr
        self.B1 -= grad_B1 * lr
        self.B2 -= grad_B2 * lr
        self.B3 -= grad_B3 * lr


nn = NeuralNetwork()
result = nn.calc_loss([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
print(result)

for i in range(1000):
    v1 = np.random.random_integers(0, 1)
    v2 = np.random.random_integers(0, 1)
    v3 = np.random.random_integers(0, 1)
    n = v1 * 4 + v2 * 2 + v3
    n_arr = np.zeros(8)
    n_arr[n] = 1
    nn.update_gradient([v1, v2, v3], n_arr)
    # nn.update_gradient([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])

result = nn.calc_loss([1, 0, 1], [0, 0, 0, 0, 1, 0, 0, 0])
print(result)

print(str(nn.predict([1, 0, 0])) + ":" + str(np.argmax(nn.predict([1, 0, 0]))))

# v1 = np.random.random_integers(0, 1)
# v2 = np.random.random_integers(0, 1)
# v3 = np.random.random_integers(0, 1)
# n = v1 * 4 + v2 * 2 + v3
# n_arr = np.zeros(8)
# n_arr[n] = 1
# print([v1, v2, v3], n_arr)
