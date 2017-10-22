# let it be library for solving machine learning problems

import numpy as np
import matplotlib.pyplot as plt
import csv

def solve(X, y):
    C = np.dot(np.transpose(X),X)
    G = np.linalg.inv(C)

    B = np.dot(np.dot(G, np.transpose(X)),y)
    return B

def show(X, Y, B, dims=[1]):
    for dim in dims:
        # Warning! It works only for 1D of x
        plt.figure()
        # show train data first 
        plt.scatter(X[:,dim], y=Y)
        # then plot prediction line
        X_line, Y_line = X[:, 1], np.dot(X,B)
        xs = [X_line[0], X_line[-1]]
        ys = [Y_line[0], Y_line[-1]]
        plt.plot(xs, ys)
        plt.show()

def normalize(X):
    # count max
    multiplier = [max(factor) for factor in X]
    new_X = []
    for i, factor in enumerate(X):
        # divide every entry by it
        new_X.append([x_i/multiplier[i] for x_i in factor])
    new_X = np.transpose(new_X)
    return new_X, multiplier

def open_csv(filename, delimiter = ","):
    with open(filename) as fn:
        data = csv.reader(fn, delimiter=delimiter)
        titles = next(data)
        features = []
        for line in data:
            features.append(line)
    return features, titles

if __name__ == "__main__":
    X = [[1, 2, 3, 4],
         [2, 3, 5, 8],
         [4, 7, 14, 1]]
    print(normalize(X))
