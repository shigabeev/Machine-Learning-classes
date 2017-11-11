import numpy as np
import math
import random

def neuron_output(inputs, weights):
    return sigmoid(np.dot(inputs, weights))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def der_sigmoid(x):
    return sigmoid(-x)*(1-sigmoid(-x))

def step_function(x):
    return 1 if x>0 else 0

def feed_forward(neural_network, input_vector):
    """takes in a neural network (represented as a list of lists of lists of weights)
    and returns the output from forward-propagating the input"""

    outputs = []

    for layer in neural_network:

        input_with_bias = list(input_vector) + [1., ]            # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                  for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target, lrate = 0.01):

    hidden_outputs, outputs = feed_forward(network, input_vector) # really, only 2 layers !

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output * lrate

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      np.dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input * lrate

def predict(network, input):
    result = feed_forward(network, input)[-1]
    return result.index(max(result))

def generate_network(input_size, num_hidden, output_size):
    # each hidden neuron has one weight per input, plus a bias weight
    hidden_layer = [[random.random() for __ in range(input_size + 1)]
                    for __ in range(num_hidden)]

    # each output neuron has one weight per hidden neuron, plus a bias weight
    output_layer = [[random.random() for __ in range(num_hidden + 1)]
                    for __ in range(output_size)]

    # the network starts out with random weights
    network = [hidden_layer, output_layer]
    return network

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    data = load_iris()
    
    network = generate_network(4, 8, 3)

    # Learning stage

    for _ in range(1000):       # 1000 epochs seems enough to converge
        for i, inputs in enumerate(data.data[:]):
            backpropagate(network, inputs, np.eye(3)[data.target[i]])

    # Testing
    for i, inputs in enumerate(data.data[:]):
        print("Predicted: ", predict(network, inputs), "Target: ", data.target[i])
   
   
    # XOR example
   
    # in1, in2 = int(input()), int(input()) # inputs for our neural network

    # weights for network

    network = generate_network(2, 3, 2)

    for i in range(1000):
        for x in [0, 1]:
            for y in [0, 1]:
                backpropagate(network, [x, y], [1, 0] if x==y else [0, 1])

    for x in [0, 1]:
        for y in [0, 1]:
            print("[%s, %s]" %(x, y), predict(network, [x, y]))
