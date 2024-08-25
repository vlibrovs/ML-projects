import numpy as np
import data_extractor as dext
from network import Network
from math import sqrt

de = dext.DataExtractor()
tr_data, tr_labels, test_data, test_labels = de.extract()

tr_data = tr_data / 255
test_data = test_data / 255

def preload():
    with open("saved/log.txt") as log:
        layers = [int(i) for i in log.readline()[1:-2].split(",")]
        best = float(log.readline())
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            weights.append(np.load(f"saved/w{i}.npy"))
            biases.append(np.load(f"saved/b{i}.npy"))

        return Network(layers, weights, biases, best)

layers = [784, 30, 10]
weights = []
biases = []
# for prev, this in zip(layers[:-1], layers[1:]):
#     lower, upper = -(1 / sqrt(prev)), (1 / sqrt(prev))
#     weights.append(lower + np.random.rand(this, prev) * (upper - lower))

# net = Network(layers, best=69.85)
net = preload()
# net.gradient_descent(tr_data, tr_labels, 30, 10, 1, test_data, test_labels)

