import numpy as np


class FullyConnected:
    """
    A fully connected layer with nin inputs and nout outputs. The weight matrix
    has shape (1 + nin, nout) to include a bias term. The output is calculated
    as x' = x @ W where x is some N x nin matrix.
    """

    def __init__(self, nin, nout):
        # Initialize weight matrix
        self.W = np.random.normal(0, np.sqrt(1.0 / nin), (1 + nin, nout))

    def forward(self, X):
        # Concatenate for bias terms, cache input, compute output
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
        self.cached_input = X
        return X.dot(self.W)

    def backward(self, grad, lr):
        # Use d(X @ W) / dW for weight updates using vanilla gradient descent
        update = np.zeros_like(self.W)
        for i in range(self.W.shape[1]):
            update[:, i] = np.mean(
                grad[:, (i, )] * self.cached_input, axis=0, keepdims=True
            )
        self.W -= lr * update

        # Use d(X @ W) / dX to pass onto later layers
        new_grad = grad.dot(self.W[1:,].T)
        return new_grad


class ReLU:
    def forward(self, X):
        self.cached_input = X
        return np.maximum(0, X)

    def backward(self, grad, lr):
        # Gradient of ReLU is either 0 or 1 depending on the input
        local_grad = np.maximum(0, np.sign(self.cached_input))
        return grad * local_grad


class Tanh:
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def forward(self, X):
        self.cached_input = X
        return self.tanh(X)

    def backward(self, grad, lr):
        local_grad = 1 - (self.tanh(self.cached_input) ** 2)
        return grad * local_grad
