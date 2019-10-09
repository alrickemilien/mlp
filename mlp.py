# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as skmetrics

class Layer:
    """
    Represents a layer (hidden or output) in our neural network.
    """

    def __init__(self, n_input=0, activation=None, weights=None, bias=None):
        """
        :param int n_input: The number of neurons in this layer.
        :param str activation: The activation function to use (if any).
        :param weights: The layer's weights.
        :param bias: The layer's bias.
        """
        self.n_input = n_input
        self.n_output = 0

        self.weights = weights
        self.bias = bias

        self.activation = activation

        self.last_activation = None
        self.error = None
        self.delta = None

        self.dZ = None
        self.dW = None
        self.dB = None

    def activate(self, X):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """

        # X is of dimension (M, N) with M the number of rows and N number of columns/feature
        # W is of dimension (N, L) with N number of row of input and L number of columns of output
        # X.dot(W) is of dimension (M, L)
        # Bias is an array of size L
        # Add B0, B1 ... BL to each row of X.dot(W)
        Z = X.dot(self.weights) + self.bias

        self.last_activation = self._apply_activation(Z)
        return self.last_activation.copy()

    def _apply_activation(self, Z):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        if self.activation == None:
            return Z
        if self.activation == 'tanh':
            return np.tanh(Z)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        if self.activation == 'softmax':
            kw = dict(axis=1, keepdims=True)
            Zrel = Z - np.max(Z, **kw) # make every value 0 or below, as exp(0) won't overflow
            ex = np.exp(Zrel)
            return ex / np.sum(ex, **kw)
        if self.activation == 'relu':
            return 1. * (Z > 0)
        return Z

    def apply_activation_derivative(self, Z):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """

        # We use Z directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.
        if self.activation == 'tanh':
            return 1 - Z ** 2
        if self.activation == 'sigmoid':
            return Z * (1 - Z)
        if self.activation == 'relu':
            return Z * (Z > 0)
        return Z

class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self, error='cee'):
        self._layers = []
        self.error = error

    def add_layer(self, layer, weights_seed=0, bias_seed=0):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """
        self._layers.append(layer)

        if (len(self._layers) == 1): return

        previous_layer = self._layers[-2]

        previous_layer.n_output = layer.n_input
        previous_layer.activation = layer.activation

        eps = 0.5
        np.random.seed(weights_seed)
        previous_layer.weights = previous_layer.weights if previous_layer.weights is not None \
                                else np.random.rand(previous_layer.n_input, previous_layer.n_output)  * 2 * eps - eps
        
        np.random.seed(bias_seed)
        previous_layer.bias = previous_layer.bias if previous_layer.bias is not None \
                                else np.random.rand(previous_layer.n_output)  * 2 * eps - eps

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """
        A = X.copy()
        for layer in self._layers:
            A = layer.activate(A)
        return A

    def apply_softmax_error_derivate(self, y_predict, y):
        if self.error == 'cee':
            return (y_predict - y) / y.shape[0]
        return y_predict - y

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        y_predict = self.feed_forward(X)

        # Loop over the layers backward and generate deltas + errors for each layer
        # for i in reversed(range(len(self._layers))):
        for i in range(len(self._layers) - 1, -1,-1):
            layer = self._layers[i]

            """
            This is the derivate error of cross entropy error function
            This error function is convinient because it's chained final derivate
            leads to a very simple formula without jacobian matrix complications
            """
            layer.dZ = self.apply_softmax_error_derivate(y_predict, y) if i == (len(self._layers) - 1) \
                else layer.apply_activation_derivative(layer.last_activation) * self._layers[i + 1].delta

            A = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            
            layer.dW = A.T.dot(layer.dZ)
            layer.dB = np.sum(layer.dZ, axis=0)

            layer.delta = layer.dZ.dot(layer.weights.T)
        
        # Gradient descent part
        for i in range(0, len(self._layers), 1):
            self._layers[i].weights -= self._layers[i].dW * learning_rate
            self._layers[i].bias -= self._layers[i].dB * learning_rate

    def train(self, X, y, X_test, y_test, learning_rate, max_epochs, batch_size=1):
        """
        Trains the neural network using backpropagation.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        :param int max_epochs: The maximum number of epochs (cycles).
        :return: The list of calculated MSE errors.
        """

        del self._layers[-1]

        mses = []
        cees = []
        for i in range(max_epochs):
        # i = 0
        # while True:
            i += 1
            from_to = np.arange(0, 1, batch_size)
            
            for j in range(0, len(from_to) - 1, 1):
                start = int(len(X) * from_to[j])
                end = int(len(X) * from_to[j + 1])
                self.backpropagation(X[start:end], y[start:end], learning_rate)
            if (from_to[-1] != 1):
                start = int(len(X) * from_to[-1])
                end = len(X)
                self.backpropagation(X[start:end], y[start:end], learning_rate)

            if i % 10 == 0:
                y_predict = self.feed_forward(X_test)
                mse = self.mean_squarred_error(y_predict, y_test)
                mses.append(mse)
                cee = self.cross_entropy_error(y_predict, y_test)
                cees.append(cee)
                print('Epoch: #%s, Batches: %d, MSE: %f, CEE: %f' % (i, int(len(X) * batch_size), float(mse), float(cee)))
        return mses, cees

    @staticmethod
    def mean_squarred_error(a, y):
        return np.mean(np.square(np.array([np.where(x == 1)[0][0] for x in y]) - np.argmax(a, axis=1)))
    
    @staticmethod
    def cross_entropy_error(a, y):
        return np.sum(-y * np.log(a)) / y.shape[0]

    def save(self, out='save.model'):
        """
        The topology of a layer is a as follow
        [layer:[activation,inputs,outputs,weights,biases]]
        """
        n = np.vstack([[x.activation, x.n_input, x.n_output, x.weights, x.bias]
                        for x in self._layers])
        np.save(out, n)

    def plot(self, mses, cees):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,8))

        axes[0].plot(np.arange(len(mses)) * 10, mses)
        axes[0].set_ylabel('MSE')
        axes[0].set_xlabel('EPOCH')

        axes[1].plot(np.arange(len(cees)) * 10, cees)
        axes[1].set_ylabel('CEE')
        axes[1].set_xlabel('EPOCH')

        plt.show()

    @staticmethod
    def accuracy(a, y):
        """
        Calculates the accuracy between the predicted labels and true labels.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The calculated accuracy.
        """
        arg = np.argmax(a, axis=1)
        pred = np.zeros(shape=y.shape)
        for xi in range(len(y)):
            pred[xi][arg[xi]] = 1
        return (pred == y).mean()
    
