# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None, weights_seed=0, bias_seed=0):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """

        eps = 0.5
        self.n_neurons = n_neurons
        self.n_input = n_input
        
        np.random.seed(weights_seed)
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)  * 2 * eps - eps
        
        np.random.seed(bias_seed)
        self.bias = bias if bias is not None else np.random.rand(n_neurons)  * 2 * eps - eps

        self.activation = activation

        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, A):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        # print('layer\'s weights', self.weights)

        r = A.dot(self.weights) + self.bias

        # print('r', r)
        # print('self.bias', self.bias)

        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        # In case no activation function was chosen
        if self.activation is None:
            return r
        # tanh
        if self.activation == 'tanh':
            return np.tanh(r)
        # sigmoid
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))
        # softamx
        if self.activation == 'softmax':
            return self.softmax(r)
        return r
    
    @staticmethod
    def softmax(r):
        ex = np.exp(r)
        return ex / ex.sum(axis=1, keepdims=True)

    @staticmethod
    def kronecker_delta(i, j):
        return 0 if i != j else 1

    def apply_activation_derivative_mse(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.

        if self.activation is None:
            return r
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            return r * (1 - r)
        if self.activation == 'softmax':
            def mse_softmax_derivate(xi):
                return [r[xi] * (r[xi] - self.kronecker_delta(xi, yi)) for (yi, _) in enumerate(r)]

            m = np.vstack([mse_softmax_derivate(xi) for (xi, _) in enumerate(r)])
            return np.dot(m, r)
        return r

class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """

        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """

        i = 0
        for layer in self._layers:
            i = i + 1
            X = layer.activate(X)
            # print('i', i)
            # if i != 4:
            #     print('X', X)
        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)
        # One row
        if ff.ndim == 1:
            return np.argmax(ff)
        # Multiple rows
        return np.argmax(ff, axis=1)

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.feed_forward(X)

        # print('output', output)

        # Loop over the layers backward and generate deltas + errors for each layer
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                """ This is the derivate error of cross entropy error function
                This error function is convinient because it's chained final derivate
                leads to a very simple formula without jacobian matrix complications
                """
                layer.delta = output - y
            else:
                next_layer = self._layers[i + 1]
                layer.error = next_layer.delta.dot(next_layer.weights.T)
                layer.delta = layer.error * layer.apply_activation_derivative_mse(layer.last_activation)
        
        # Gradient descent part
        for i in range(len(self._layers)):
            layer = self._layers[i]

            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights -= input_to_use.T.dot(layer.delta) * learning_rate
            layer.bias -= layer.delta.sum(axis=0) * learning_rate

    def train(self, X, y, learning_rate, max_epochs):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        mses = []
        cees = []
        for i in range(max_epochs):
            self.backpropagation(X, y, learning_rate)

            if i % 10 == 0:
                ff = self.feed_forward(X)
                mse = self.mean_squarred_error(ff, y)
                mses.append(mse)
                cee = self.cross_entropy_error(ff, y)
                cees.append(cee)
                print('Epoch: #%s, MSE: %f, CEE: %f' % (i, float(mse), float(cee)))
        return mses, cees

    @staticmethod
    def mean_squarred_error(a, y):
        """ This is the log loss function """
        return np.mean(np.square([np.where(x == 1)[0][0] for x in y] - np.argmax(a, axis=1)))
    
    @staticmethod
    def cross_entropy_error(a, y):
        """ This is the log loss function """
        return np.sum(-y * np.log(a))

    def save(self, out='save.model'):
        """
        The format of the save is as following
        [layer:[activation,neurons,weights,biases]]
        """
        def f(layer):
            return [layer.activation, layer.n_input, layer.n_neurons, layer.weights, layer.bias]
        np.save(out, np.vstack([f(x) for x in self._layers]))

    def plot(self, mses):
        plt.plot(np.arange(len(mses)) * 10, mses)
        plt.ylabel('MSE')
        plt.xlabel('EPOCH')
        plt.show()

    def evaluate(self, y_predict, y):
        return skmetrics.log_loss(y, y_predict) / len(y)

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        return (y_pred == y_true).mean()
    
