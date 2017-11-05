import numpy as np
import math

def neuron_output(inputs, weights, function):
    return function(np.dot(inputs, weights))

def sigmoid(x):
    return 1/(1+math.exp(-x))

def step_function(x):
    return 1 if x>0 else 0

def feed_forward(network, inputs):
    out = inputs + [1, ]  # network takes outputs from previous layer as input + bias
    for layer in XOR_nw:
        layer_output = [] 
        for weights in layer:
            layer_output.append(neuron_output(out, weights, sigmoid))
        out = layer_output + [1,] # add bias for next layer
    return out[:-1]

if __name__ == "__main__":
    in1, in2 = int(input()), int(input()) # inputs for our neural network

    # weights for network

    XOR_nw = [[[20, 20, -30],
               [20, 20, -10]],
               # output layer
               [[-60, 60, -30]]]

    print(round(feed_forward(XOR_nw, [in1, in2])[0]))

