import matplotlib.pyplot as plt
import re
import numpy as np

labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
fname = "iris.data"
epochs = 100

def s(x):
    return 1 / (1 + np.exp(-x))

def ds(x):
    return x * (1 - x)

def read_data(fn, limit, features):
    f = open(fn, "r")
    x = []
    y = []
    c = 0
    for l in f:
        if c >= limit: break
        c += 1
        l = re.split(',', l)
        if len(l) == 5:
            x.append([float(l[features[0]]), float(l[features[1]])])
            y.append(labels.index(l[-1][:-1]))
    return np.asarray(x), np.asarray(y)

#data visualization
def show_data(x,y):
    x = np.asarray(x)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

class Model:
    def __init__(self, lr):
        self.w = np.random.randn((2))
        self.b = np.random.randn()
        self.lr = lr

    def forward(self, x):
        return s(np.dot(x, self.w)+self.b)

    def backward(self, x, y_, y):
        err = (y_ - y)
        mse = np.square(err).mean()
        d_w = np.sum(x * np.expand_dims(err, axis=1), axis=0)
        self.w += - d_w * self.lr
        self.b += - err * self.lr
        return mse


x, y = read_data(fname, 100, (0,2))
# show_data(x,y)
model = Model(0.001)
for i in range(epochs):
    out = model.forward(x)
    print(model.backward(x, out, y))
