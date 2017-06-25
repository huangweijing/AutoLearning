from com.weijinglab.py.autolearning.common.NeuralNetwork import NeuralNetwork


def measure(neural_network: NeuralNetwork
            , generate_input: function
            , validate_output: function):
    good_answer_cnt = 0.0
    total_try_cnt = 1000.0

    for i in range(total_try_cnt):

        input_data = generate_input(i)
        result = neural_network.predict(input_data)

        if validate_output(input_data, result):
            good_answer_cnt = good_answer_cnt + 1

    return good_answer_cnt / total_try_cnt
