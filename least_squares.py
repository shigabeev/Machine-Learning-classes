# let it be library for solving machine learning problems

import numpy as np
import matplotlib.pyplot as plt

def solve(X, y):
    C = np.dot(np.transpose(X),X)
    G = np.linalg.inv(C)

    B = np.dot(np.dot(G, np.transpose(X)),y)
    return B

def show(X, Y, B):
    # Warning! It works only for 1D of x
    plt.figure()
    # show train data first 
    plt.scatter(X[:,1], y=Y)
    # then plot prediction line
    X_line, Y_line = X[:, 1], np.dot(X,B)
    xs = [X_line[0], X_line[-1]]
    ys = [Y_line[0], Y_line[-1]]
    plt.plot(xs, ys)
    plt.show()