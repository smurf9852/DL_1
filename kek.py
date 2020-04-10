import matplotlib.pyplot as plt
import re
import numpy as np

labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
fname = "iris.data"

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
    return x, y

#data visualization
def show_data(x,y):
    x = np.asarray(x)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()

class Model:
    def __init__(self):
        self.w = np.random.randn((2))
        self.b = np.random.randn()

    def forward(self, x):
        return s(np.dot(x, self.w)+self.b)

    def backward(self, ):
        return


x, y = read_data(fname, 100, (0,2))
# show_data(x,y)
model = Model()
print(model.forward(x))