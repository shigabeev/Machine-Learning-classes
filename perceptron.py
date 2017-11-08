import numpy as np
import math

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

        input_with_bias = input_vector + [1]             # add a bias input
        output = [neuron_output(neuron, input_with_bias) # compute the output
                  for neuron in layer]                   # for this layer
        outputs.append(output)                           # and remember it

        # the input to the next layer is the output of this one
        input_vector = output

    return outputs

def backpropagate(network, input_vector, target):

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output * (1 - output) is from the derivative of sigmoid
    output_deltas = [output * (1 - output) * (output - target[i])
                     for i, output in enumerate(outputs)]

    # adjust weights for output layer (network[-1])
    for i, output_neuron in enumerate(network[-1]):
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output

    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1 - hidden_output) *
                      np.dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input

def predict(network, input):
    result = feed_forward(network, input)[-1]
    return result.index(max(result))

if __name__ == "__main__":
    # in1, in2 = int(input()), int(input()) # inputs for our neural network

    # weights for network

    XOR_nw = [[[1, 0.5, 0.3], #n1 neuron
               [0.1, 0.2, 1]],#n2 neuron
               # output layer
               [[1, 0.5, 0.3], #n1 neuron
               [0.1, 0.2, 1]]  #o2 neuron
               ]

    for i in range(1000):
        for x in [0, 1]:
            for y in [0, 1]:
                backpropagate(XOR_nw, [x, y], [1, 0] if x==y else [0, 1])

    for x in [0, 1]:
        for y in [0, 1]:
            print("[%s, %s]" %(x, y), predict(XOR_nw, [x, y]))
