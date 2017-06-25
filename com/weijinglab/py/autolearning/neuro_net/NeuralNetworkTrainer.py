from com.weijinglab.py.autolearning.common.NeuralNetwork import NeuralNetwork
import numpy as np


class NeuralNetworkTrainer:
    __neural_network = NeuralNetwork(input_size=1, output_size=2)

    pass


def measure(nn: NeuralNetwork):
    correct_cnt = 0.0
    total_cnt = 10000.0

    for i in range(int(total_cnt)):
        input_data = np.random.randint(1, 10000)
        result = np.argmax(nn.predict(input_data))
        if result == input_data % 2:
            correct_cnt = correct_cnt + 1

    return correct_cnt / total_cnt


def aaa():
    nn = NeuralNetwork(input_size=1, output_size=2, hidden_layer_size=[4])
    for i in range(10000):
        input_data = np.random.randint(1, 10000)
        output_data = np.zeros(2)
        output_data[input_data % 2] = 1
        nn.update_gradient([input_data], output_data)
        if i % 1000 == 0:
            print(measure(nn))

    print(np.argmax(nn.predict([40])))
    print(measure(nn))


def generate_input(num):
    return np.random.randint(1, 10000)

def valid_output(input_data, result_data):
    output_data = np.zeros(2)
    output_data[input_data % 2] = 1

    outp

    return output_data




def bbb():
    input_data = 100  # np.random.randint(1, 10000)
    output_data = np.zeros(2)
    output_data[input_data % 2] = 1

    nn = NeuralNetwork(input_size=1, output_size=2, hidden_layer_size=[16, 8, 4])
    print("input_data = %s, result = %s" % (str(input_data), str(nn.predict(input_data))))
    # print(measure(nn))
    for i in range(100):
        n = i
        output_data = np.zeros(2)
        output_data[n % 2] = 1
        nn.update_gradient([n], output_data)

    input_data = 43  # np.random.randint(1, 10000)
    print(input_data)
    output_data = np.zeros(2)
    output_data[input_data % 2] = 1

    print("input_data = %s, result = %s" % (str(input_data), str(nn.predict(input_data))))


bbb()


# print(4%2)
# test()
# print(np.exp(-1000800))
