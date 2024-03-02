class Model:
    def __init__(self, layers, loss, learning_rate):
        self.layers = layers
        self.loss = loss
        self.learning_rate = learning_rate

    def predict(self, X):
        # Pass the input forward through every layer
        ret = X
        for layer in self.layers:
            ret = layer.forward(ret)
        return ret

    def update(self, pred, y):
        # Pass the gradient backwards through every layer
        loss = self.loss.forward(pred, y)
        grad = self.loss.backward()

        for layer in reversed(self.layers):
            grad = layer.backward(grad, self.learning_rate)
        return loss

    def train(self, X, y):
        # Train once on the data X, y
        y_pred = self.predict(X)
        loss = self.update(y_pred, y)
        return loss
