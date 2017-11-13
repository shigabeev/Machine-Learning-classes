import numpy as np
import math
import random

class Perceptron:
    def sigmoid(x, derivative = False):
        if not derivative:
            return 1/(1+math.exp(-x))
        return sigmoid(-x)*(1-sigmoid(-x))

    def neuron_output(self, inputs, weights, activation = sigmoid):
        return activation(np.dot(inputs, weights))

    def feed_forward(self, input_vector):
        """takes in a neural network (represented as a list of lists of lists of weights)
        and returns the output from forward-propagating the input"""

        outputs = []

        for layer in self.network:

            input_with_bias = np.append(input_vector, 1)            # add a bias input
            output = [self.neuron_output(neuron, input_with_bias) # compute the output
                    for neuron in layer]                   # for this layer
            outputs.append(output)                           # and remember it

            # the input to the next layer is the output of this one
            input_vector = output

        return outputs

    def backpropagate(self, input_vector, target):

        hidden_outputs, outputs = self.feed_forward(input_vector = input_vector) # really, only 2 layers !

        # the output * (1 - output) is from the derivative of sigmoid
        output_deltas = [output * (1 - output) * (output - target[i])
                        for i, output in enumerate(outputs)]

        # adjust weights for output layer (network[-1])
        for i, output_neuron in enumerate(self.network[-1]):
            for j, hidden_output in enumerate(hidden_outputs + [1]):
                output_neuron[j] -= output_deltas[i] * hidden_output * self.lrate

        # back-propagate errors to hidden layer
        hidden_deltas = [hidden_output * (1 - hidden_output) *
                        np.dot(output_deltas, [n[i] for n in self.network[-1]])
                        for i, hidden_output in enumerate(hidden_outputs)]

        # adjust weights for hidden layer (network[0])
        for i, hidden_neuron in enumerate(self.network[0]):
            for j, input in enumerate(input_vector + [1]):
                hidden_neuron[j] -= hidden_deltas[i] * input * self.lrate
    
    def __init__(self, activation=sigmoid, num_hidden=8, lrate=0.01):
        self.activation=activation
        self.lrate = lrate
        self.num_hidden = num_hidden
    
    def fit(self, X, y, epochs = 100, warm_start = False, shuffle=False):
        if not warm_start:
            self.no_outs = max(y) + 1
            self.generate_network(len(X[0]), self.num_hidden, self.no_outs)
        for _ in range(epochs):       # 1000 epochs seems enough to converge
            for i, inputs in enumerate(X):
                self.backpropagate(input_vector=inputs,target = np.eye(self.no_outs)[y[i]])
    
    def generate_network(self, input_size, num_hidden, output_size):
        # each hidden neuron has one weight per input, plus a bias weight
        hidden_layer = [[random.random() for __ in range(input_size + 1)]
                        for __ in range(num_hidden)]

        # each output neuron has one weight per hidden neuron, plus a bias weight
        output_layer = [[random.random() for __ in range(num_hidden + 1)]
                        for __ in range(output_size)]

        # the network starts out with random weights
        self.network = [hidden_layer, output_layer]

    def predict(self, input):
        result = self.feed_forward(input)[-1]
        return result.index(max(result))

def generate_data(X, Y, ratio = 0.9):
    corpus = [[X[i]]+[Y[i]] for i, _ in enumerate(Y)]
    random.shuffle(corpus)
    train_amount = int(len(corpus)*ratio)
    train_target = [corpus[i][1] for i in range(train_amount)]
    train_data = [corpus[i][0] for i in range(train_amount)]
    test_target = [corpus[i][1] for i in range(train_amount, len(corpus))]
    test_data = [corpus[i][0] for i in range(train_amount, len(corpus))]
    return train_data, train_target, test_data, test_target

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    import matplotlib.pyplot as plt

    iris_dataset = load_iris()
    
    train_data, train_target, test_data, test_target = generate_data(
        iris_dataset.data, iris_dataset.target
    )


    clf = Perceptron(num_hidden=8, lrate = 0.01)
    clf.fit(train_data, train_target, epochs = 1000)
    
    misses = 0
    matches = 0
    for i, inputs in enumerate(test_data):
        predicted = clf.predict(inputs)
        target = test_target[i]
        print("Predicted: ", predicted, "Target: ", target)
        if predicted == target:
            matches +=1
        else:
            misses += 1
    accuracy = matches/(misses+matches)
    print ("Classifier accuracy : " + "{:.2%}".format(accuracy))
    if accuracy >= .98:
        print("Excellent !")
    elif accuracy >= .90:
        print("Good")
    elif accuracy > .75:
        print("Average")
    else:
        print("Fair")