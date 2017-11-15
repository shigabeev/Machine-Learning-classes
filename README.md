# Neural networks
There are materials for «special math» classes in NUST MISiS.
Production code located in .py files.
All the research is inside jupyter notebooks.
## Least squares
First task was to implement least squares regression.

![least squares](https://sun9-8.userapi.com/c639325/v639325010/5c519/_xCjDq7F7nQ.jpg)

### Usage:
To run sample and see what it does, run following 
     
    $ python3 least_squares.py
    
In order to use it as a black box, add python file to your working directory and import it.

    import least_squares as ls
Now you can fit/predict it as usual.

    ols = ls.LeastSquares(constant = True)
    ols.fit(X, y)
    ols.predict(x)
## Perceptron
Second task was to implement Perceptron.
Perceptron has one hidden layer and sigmoid activation function.
Weights adjust automatically, using back-propagation.
### Usage
Running

    $ python3 perceptron.py
Will show you how it performs on iris dataset.

To use it in your project:

1.clone it
    
    $ curl -o perceptron.py https://github.com/shigabeev/neural_networks/blob/master/perceptron.py
2.import

    import perceptron
3.fit/ predict
    
    clf = perceptron.Perceptron()
    clf.fit(X, y)
    clf.predict(x)
