import matplotlib.pyplot as plt
import re
import numpy as np

labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
fname = "monk2.csv"
epochs = 10000

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
        if len(l) >= 5:
            # x.append([float(l[features[0]]), float(l[features[1]])])
            x.append([float(x) for x in l[:-1]])
            y.append(int(l[-1][:-1]))
            # y.append(labels.index(l[-1][:-1]))
    return np.asarray(x), np.asarray(y)

#data visualization
def show_data(x,y):
    x = np.asarray(x)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

class Model:
    def __init__(self, f):
        self.w = np.random.randn((f))
        self.b = np.random.randn()

    def forward(self, x):
        return s(np.dot(x, self.w)+self.b)

    def backward(self, x, y_, y, lr, reg):
        err = (y_ - y)
        mse = np.square(err).mean()
        d_w = np.sum(x * np.expand_dims(err, axis=1), axis=0)
        self.w *= 1 - reg
        self.b *= 1 - reg
        self.w += - d_w * lr
        self.b += - np.sum(err) * lr
        return mse

    def reset(self, f):
        self.w = np.random.randn((f))
        self.b = np.random.randn()

x_all, y_all = read_data(fname, 432, (0,2))
x_train, y_train = x_all[:344], y_all[:344]
x_test, y_test = x_all[344:], y_all[344:]
# show_data(x,y)
model = Model(6)

for lr in [0.001, 0.0005]:
    mses = []
    for i in range(epochs):
        out = model.forward(x_train)
        model.backward(x_train, out, y_train, lr, 0)
        test_loss = np.square(model.forward(x_test) - y_test).mean()
        mses.append(test_loss)

    plt.plot(range(len(mses)), mses, label=lr)
    model.reset(6)

plt.legend()
plt.show()