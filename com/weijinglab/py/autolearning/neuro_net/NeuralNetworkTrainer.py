from com.weijinglab.py.autolearning.common.NeuralNetwork import NeuralNetwork
import numpy as np

class NeuralNetworkTrainer:
    __neural_network = NeuralNetwork(input_size=1, output_size=2)

    pass




def test():
    nn = NeuralNetwork(input_size=1, output_size=2)
    for i in range(1000):
        input_data = np.random.randint(1, 10000)
        output_data = np.zeros(2)
        output_data[input_data % 2] = 1
        nn.update_gradient([input_data], output_data)

    print(np.argmax(nn.predict([40])))

# print(4%2)
test()
# print(np.exp(-1000800))