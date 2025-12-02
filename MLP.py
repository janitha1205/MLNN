import numpy as np
import matplotlib.pyplot as plt


# Creating labels
y = [[1], [1], [1], [0]]
# visualizing the data, plotting A.


# converting data and labels into numpy array
x = [np.array([[0, 0]]), np.array([[1, 0]]), np.array([[0, 1]]), np.array([[1, 1]])]
y = np.array(y)
# Printing data and labels
print(x, "\n\n", y)


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(z):
    return np.maximum(0, z)


def relu_derivative(A):

    return (A > 0) * 1.0


# Creating the Feed forward neural network
def f_forward(x, w1, w2, w3):
    # hidden
    z1 = x.dot(w1)  # input from layer 1
    a1 = sigmoid(z1)  # out put of layer 2
    z2 = a1.dot(w2)  # input of out layer
    a2 = relu(z2)  # output of out layer
    z3 = a2.dot(w3)
    a3 = relu(z3)
    return a3


# initializing the weights randomly
def generate_wt(x, y):
    li = []
    for i in range(x * y):
        li.append(np.random.randn())
    return np.array(li).reshape(x, y)


# for loss we will be using mean square error(MSE)
def loss(out, Y):
    s = np.square(out - Y)
    s = np.sum(s) / len(y)
    return s


# Back propagation of error
def back_prop(x, y, w1, w2, w3, alpha):

    # hidden layer
    z1 = x.dot(w1)
    a1 = sigmoid(z1)
    z2 = a1.dot(w2)
    a2 = relu(z2)
    z3 = a2.dot(w3)
    a3 = relu(z3)

    # error in output layer
    d3 = a3 - y
    print(w3.shape)
    d2 = np.multiply((w3.dot((d3.transpose()))).transpose(), relu_derivative(a2))
    print(w2.shape)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), (np.multiply(a1, 1 - a1)))
    # Gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
    w3_adj = a2.transpose().dot(d3)

    # Updating parameters
    w1 = w1 - (alpha * (w1_adj))
    w2 = w2 - (alpha * (w2_adj))
    w3 = w3 - (alpha * (w3_adj))

    return (w1, w2, w3)


w1 = generate_wt(2, 5)
w2 = generate_wt(5, 5)
w3 = generate_wt(5, 1)
print(w1, "\n\n", w2, "\n\n", w3)


def train(x, Y, w1, w2, w3, alpha=0.01, epoch=10):
    acc = []
    losss = []
    for j in range(epoch):
        l = []
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2, w3)
            l.append((loss(out, Y[i])))
            w1, w2, w3 = back_prop(x[i], y[i], w1, w2, w3, alpha)
        print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
        acc.append((1 - (sum(l) / len(x))) * 100)
        losss.append(sum(l) / len(x))
    return (acc, losss, w1, w2, w3)


acc, losss, w1, w2, w3 = train(x, y, w1, w2, w3, 0.1, 100)


# plotting accuracy
plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs:")
plt.show()

# plotting Loss
plt.plot(losss)
plt.ylabel("Loss")
plt.xlabel("Epochs:")
plt.show()


def predict(x, w1, w2, w3):
    Out = f_forward(x, w1, w2, w3)
    print(Out)


# Example: Predicting for letter 'B'
predict(x[0], w1, w2, w3)
predict(x[1], w1, w2, w3)
predict(x[2], w1, w2, w3)
predict(x[3], w1, w2, w3)
