import numpy as np

class LinearRegressionModel:
    def __init__(self, weight=1, bias=1):
        self.weight = weight
        self.bias = bias

    def predict(self, x):
        return self.weight * x + self.bias
    
    def __cost(self, x, y):
        return (y - self.predict(x)) ** 2
    
    def train(self, x: np.ndarray, y: np.ndarray, epochs=1, rate=1.0):
        n = len(x)
        for _ in range(epochs):
            del_b = 2 * (self.predict(x) - y)
            del_w = del_b * x
            del_b = np.sum(del_b) / n
            del_w = np.sum(del_w) / n
            self.weight -= rate * del_w
            self.bias -= rate * del_b

