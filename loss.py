import numpy as np


class MSELoss:
    def forward(self, y_pred, y):
        self.cached_input = y_pred, y
        return 0.5 * np.mean((y_pred - y) ** 2)

    def backward(self):
        y_pred, y = self.cached_input
        return (y_pred - y) / len(y)
