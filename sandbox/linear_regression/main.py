import numpy as np
import matplotlib.pyplot as plt
from linreg import LinearRegressionModel

size = 1000
x = np.random.rand(size) * 10
y = 2 * x + 5 * np.random.randn(size) + 2

model = LinearRegressionModel()
model.train(x, y, 1000, 0.01)

x0 = np.array([np.min(x), np.max(x)])
y0 = model.predict(x0)

plt.plot(x, y, "o")
plt.plot(x0, y0)
plt.show()