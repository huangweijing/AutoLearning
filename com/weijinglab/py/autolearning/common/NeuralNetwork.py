import numpy as np

from com.weijinglab.py.autolearning.common import LearningFuncs


class NeuralNetwork:
    # X = np.zeros_like(3)
    # W1 = np.zeros_like(3, 6)
    # B1 = np.zeros_like(6)
    # W2 = np.zeros_like(6, 12)
    # B2 = np.zeros_like(12)
    # Y = np.zeros_like(8)
    #LearningFuncs.sigmoid(X)

    def __init__(self, input_size=3, output_size=8, hidden_layer_size=[6, 12]):
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
        output_data = LearningFuncs.sigmoid(output_data)
        output_data = LearningFuncs.softmax(output_data)

        return output_data


nn = NeuralNetwork()
print(nn.predict(np.array([1, 0, 1])))
# print(nn.B1)