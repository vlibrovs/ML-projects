import numpy as np
import random

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Network():
    def __init__(self, layers, weights=None, biases=None, best=0.0):
        self.layers = layers
        self.weights = []
        self.biases = []
        if weights is None: self.weights = [np.random.randn(i, j) for i, j in zip(layers[1:], layers[:-1])]
        else: self.weights = weights
        # self.biases = [np.zeros(i) for i in layers[1:]]
        if biases is None: self.biases = [np.random.randn(i) for i in layers[1:]]
        else: self.biases = biases
        self.best = best

    def save_params(self):
        with open("saved/log.txt", "w") as log:
                log.write(str(self.layers))
                log.write("\n")
                log.write(str(self.best))
        for index, (w, b) in enumerate(zip(self.weights, self.biases)):
            np.save(f"saved/w{index}.npy", w)
            np.save(f"saved/b{index}.npy", b)
            

    def run(self, inp):
        for w, b in zip(self.weights, self.biases):
            inp = w @ inp + b
        return np.argmax(inp)

    def test(self, data, labels):
        correct = 0
        for obj, label in zip(data, labels):
            correct += (self.run(obj) == label)
        print(f"Test results: {correct}/{len(data)}, accuracy: {100 * correct / len(data)}%")
        return 100 * correct / len(data)

    def gradient_descent(self, tr_data, labels, epochs, minibatch_size, rate, test_data=None, test_labels=None):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            shuffled_data = [(data, label) for data, label in zip(tr_data, labels)] 
            random.shuffle(shuffled_data)
            minibatches = []
            for i in range(0, len(tr_data), minibatch_size):
                minibatches.append(shuffled_data[i:i + minibatch_size])
            for minibatch in minibatches:
                self.update(minibatch, rate)

            if not (test_data is None or test_labels is None):
                result = self.test(test_data, test_labels)
                if result > self.best:
                    self.best = result
                    self.save_params()

    def update(self, minibatch, rate):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for data, label in minibatch:
            expected = np.zeros(self.layers[-1])
            expected[label] = 1
            dw, db = self.backprop(data, expected)
            nabla_w = [w0 + dw0 for w0, dw0 in zip(nabla_w, dw)]
            nabla_b = [b0 + db0 for b0, db0 in zip(nabla_b, db)]

        self.weights = [w - (rate / len(minibatch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (rate / len(minibatch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, data, expected):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Forward pass

        a = data
        activ = [data]
        zs = []

        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            activ.append(a)

        # Backward pass

        activation_der = 2 * (activ[-1] - expected) * sigmoid_derivative(zs[-1])

        for l in range(1, len(self.layers)):
            nabla_w[-l] += np.array([activ[-l-1] * ad for ad in activation_der])
            nabla_b[-l] += activation_der

            activation_der = self.weights[-l].T @ activation_der

        return nabla_w, nabla_b

            
        