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
        # This leads to matrix of size (n_input, n_neurons)
        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)  * 2 * eps - eps
        
        np.random.seed(bias_seed)
        self.bias = bias if bias is not None else np.random.rand(n_neurons)  * 2 * eps - eps

        self.activation = activation

        self.last_activation = None
        self.error = None
        self.delta = None

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
        return self.last_activation

    def _apply_activation(self, Z):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The "activated" value.
        """

        if self.activation == 'tanh':
            return np.tanh(Z)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        if self.activation == 'softmax':
            kw = dict(axis=1, keepdims=True)

            # make every value 0 or below, as exp(0) won't overflow
            Zrel = Z - Z.max(**kw)

            # if you wanted better handling of small exponents, you could do something like this
            # to try and make the values as large as possible without overflowing, The 0.9
            # is a fudge factor to try and ignore rounding errors
            #
            #     xrel += np.log(np.finfo(float).max / x.shape[axis]) * 0.9

            ex = np.exp(Zrel)
            return ex / ex.sum(**kw)  
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
        return Z

class NeuralNetwork:
    """
    Represents a neural network.
    """

    def __init__(self, error='cee'):
        self._layers = []
        self.error = error

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

        tmp = X
        for layer in self._layers:
            tmp = layer.activate(tmp)
        return tmp

    @staticmethod
    def kronecker_delta(i, j):
        return 0 if i != j else 1
    
    def apply_error_derivate(self, out, y):
        if self.error == 'cee':
            return out - y
        if self.error == 'mse':
            m = np.array([
                [ out.T[xi].sum() * (out.T[yi].sum() - self.kronecker_delta(xi, yi)) for (yi, _) in enumerate(out.T)]
                for (xi, _) in enumerate(out.T)])
            return out.dot(m)
        return out - y

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
                layer.delta = self.apply_error_derivate(output, y)
            else:
                next_layer = self._layers[i + 1]
                layer.error = next_layer.delta.dot(next_layer.weights.T)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)
        
        # Gradient descent part
        for i in range(len(self._layers)):
            layer = self._layers[i]

            # The input is either the previous layers output or X itself (for the first hidden layer)
            A = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)

            layer.weights -= A.T.dot(layer.delta) * learning_rate
            layer.bias -= layer.delta.sum(axis=0) * learning_rate

    def train(self, X, y, X_test, y_test, learning_rate, max_epochs):
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
        # i = 0
        # while i:
            i += 1
            self.backpropagation(X, y, learning_rate)

            if i % 10 == 0:
                y_predict = self.feed_forward(X_test)
                mse = self.mean_squarred_error(y_predict, y_test)
                mses.append(mse)
                cee = self.cross_entropy_error(y_predict, y_test)
                cees.append(cee)
                print('Epoch: #%s, MSE: %f, CEE: %f' % (i, float(mse), float(cee)))
        return mses, cees

    @staticmethod
    def mean_squarred_error(a, y):
        return np.mean(np.square(np.array([np.where(x == 1)[0][0] for x in y]) - np.argmax(a, axis=1)))
    
    @staticmethod
    def cross_entropy_error(a, y):
        return skmetrics.log_loss(y, a)
        """ This is the log loss function """
        return np.sum(-y * np.log(a))

    def save(self, out='save.model'):
        """
        The format of the save is as following
        [layer:[activation,neurons,weights,biases]]
        """
        n = np.vstack([[x.activation, x.n_input, x.n_neurons, x.weights, x.bias]
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
    
