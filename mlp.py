# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None, seed=0):
        """
        :param int n_input: The input size (coming from the input layer or a previous hidden layer)
        :param int n_neurons: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """

        np.random.seed(seed)

        eps = 0.5
        self.n_neurons = n_neurons
        self.n_input = n_input
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)  * 2 * eps - eps
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)  * 2 * eps - eps
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        # print('layer\'s weights', self.weights)

        r = np.dot(x, self.weights) + self.bias

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
            ex = np.exp(r)
            return ex / ex.sum(axis=0, keepdims=True)
        return r
    
    def apply_activation_derivative(self, r):
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
            return r * (1 - r)
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

        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """

        ff = self.feed_forward(X)
        # print('ff', ff)
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

        # print('backpropagation', X)
        # print('OUTPUT', output)

        # Loop over the layers backward and generate deltas + errors for each layer
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            # If this is the output layer
            if layer == self._layers[-1]:
                layer.error = y - output
                # The output is layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights of each layer
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

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
        bces = []
        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)
            if i % 10 == 0:
                ff = self.feed_forward(X)
                mse = np.mean(np.square([np.where(x == 1)[0][0] for x in y] - np.argmax(ff, axis=1)))
                mses.append(mse)
                bce = np.mean(skmetrics.log_loss(y, ff))
                bces.append(bce)
                print('Epoch: #%s, MSE: %f, BCE: %f' % (i, float(mse), float(bce)))
        return mses, bces

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
        # print('y_predict', y_predict)
        return skmetrics.log_loss(y, y_predict)
        size = np.size(y_predict, 0)
        y_predict = y_predict.reshape(-1, 2)[:, 0]
        y = y[:, 0]
        return ((1 / size)
            * (-1 * y.dot(np.log(y_predict))
            - (1 - y).dot(np.log(1 - y_predict))))

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        return (y_pred == y_true).mean()
    
