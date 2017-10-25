# let it be library for solving machine learning problems

import numpy as np
import matplotlib.pyplot as plt
import csv

class LeastSquares:
    def __init__(self, constant=False, pow_2_factors = None):
        self.constant = constant
        self.pow_2_factors = pow_2_factors
        pass
    
    def fit(self, X, y):
        '''
        Fit Least Squares regression model 
        
        using formula B = (X^T * X)^(-1) * X^T * Y

        Parameters
        ----------
        X - matrix of features of size (m x n), [[x_11, x_21, ... , x_m1]
                                                 [x_12, x_22, ... , x_m2]
                                                              ...
                                                 [x_1n, x_2n, ... , x_mn]]
        y - array of true outputs
        
        Returns
        -------
        self : returns an instance of self.
        '''
        self.X=X
        self.y=y
        if self.constant:
            X = self.add_constant(X)
        for factor in self.pow_2_factors:
            X = self.add_squared(X, factor)
        C = np.dot(np.transpose(X), X)
        G = np.linalg.inv(C)

        self.B = np.dot(np.dot(G, np.transpose(X)),y)

    def predict(self, X):
        '''
        Predicts outputs based on X and model
        '''
        Y = []
        if isinstance(X[0], list):
            y = 0
            for line in X:
                for i, x in enumerate(line):
                    y = y + x*self.B[i]
            Y.append(y)
        else:
            Y = [x*self.B[0] for x in X]
        return Y

    def solve_(self, X, y):
        
        C = np.dot(np.transpose(X),X)
        G = np.linalg.inv(C)

        B = np.dot(np.dot(G, np.transpose(X)),y)
        return B

    def add_constant(self, X):
        '''
        Adds row of ones to matrix X
        '''
        if isinstance(X[0], list):
            return [[1.] + x for x in X]
        else:
            return [[1, x] for x in X]

    def add_squared(self, X, factor):
        '''
        Adds square of factor to matrix X
        '''
        print(X[0])
        if isinstance(X[0], list):
            return [[float(k) for k in x[:]] + x[factor]**2 for x in X]
        else:
            return [[x, x**2] for x in X]
        
def plot_errors(Y_True, Y_pred):
    E = []
    assert(Y_pred == Y_pred)
    for i, _ in enumerate(Y_True):
        E.append(Y_True[i]-Y_pred[i])
    plt.figure()
    plt.scatter(x = Y_True, y=E)
    plt.show()
    return E

def solve(X, y):
    C = np.dot(np.transpose(X),X)
    G = np.linalg.inv(C)

    B = np.dot(np.dot(G, np.transpose(X)),y)
    return B

def generate_data(X, Y, ratio = 0.9):
    '''
    Splits data in two according to ratio into train data and test data
    '''
    from random import shuffle
    corpus = [[X[i]]+[Y[i]] for i, _ in enumerate(Y)]
    shuffle(corpus)
    train_amount = int(len(corpus)*ratio)
    train_target = [corpus[i][1] for i in range(train_amount)]
    train_data = [corpus[i][0] for i in range(train_amount)]
    test_target = [corpus[i][1] for i in range(train_amount, len(corpus))]
    test_data = [corpus[i][0] for i in range(train_amount, len(corpus))]
    return train_data, train_target, test_data, test_target


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
    # one-dimensional
    features, titles = open_csv("car_price/data.csv", delimiter=';')
    X = [float(x[0]) for x in features]
    Y = [float(x[-1]) for x in features]
    del features, titles
    
    train_data, train_target, test_data, test_target = generate_data(X, Y)

    clf = LeastSquares(constant=True, pow_2_factors=[])
    clf.fit(train_data, train_target)
    
    plot_errors(test_target, clf.predict(test_data))

    '''
    features, titles = open_csv("car_price/data.csv", delimiter=';')
    X = [[1.] + [float(k) for k in x[:5]] + [float(x[1])**2] for x in features]  # + [float(x[7])]
    Y = [float(x[-1]) for x in features]

    train_data, train_target, test_data, test_target = generate_data(X, Y)

    clf = LeastSquares()
    clf.fit(train_data, train_target)
    
    plot_errors(test_target, clf.predict(test_data))
    '''
