import numpy as np
import pandas as pd


class Activation_ReLu:
    def forward(self, z):
        return np.maximum(z, 0)

    def backward(self, z):
        return np.where(z > 0, 1, 0)


class Activation_Softmax:
    def forward(self, z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        norm = exp / np.sum(exp, axis=1, keepdims=True)

        return norm


class Activation_Linear:
    def forward(self, z):
        return z


activation1 = Activation_ReLu()
activation2 = Activation_Softmax()
activation3 = Activation_Linear()

class NeuralNetwork:
    def __init__(self, input_size, layer_size, output_size, loss='mse'):
        self.w1 = np.random.randn(input_size, layer_size)
        self.b1 = np.zeros((1, layer_size))
        self.w2 = np.random.randn(layer_size, output_size)
        self.b2 = np.zeros((1, output_size))

        self.loss = loss

    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = activation1.forward(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = activation3.forward(self.z2)

        return self.a2

    def backward(self, X, y, y_pred, l2_lambda=0.0):
        m = X.shape[0]

        self.dz2 = (y_pred - y) / m
        self.dw2 = self.a1.T @ self.dz2 + l2_lambda * self.w2
        self.db2 = self.dz2.sum(axis=0, keepdims=True)

        dz1 = (self.dz2 @ self.w2.T) * activation1.backward(self.z1)
        self.dw1 = X.T @ dz1 / m + l2_lambda * self.w1
        self.db1 = dz1.sum(axis=0, keepdims=True) / m

    def update_parameters(self, learning_rate):
        self.w1 -= self.dw1 * learning_rate
        self.b1 -= self.db1 * learning_rate
        self.w2 -= self.dw2 * learning_rate
        self.b2 -= self.db2 * learning_rate

    def calculate_loss(self, y_pred, y, l2_lambda=0.0):
        if self.loss == 'cce':
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
            p_true = (y_pred * y).sum(axis=1)
            data_loss = -np.mean(np.log(p_true))

            # L2 penalty (exclude biases!)
            l2_loss = (l2_lambda / 2) * (np.sum(self.w1 ** 2) + np.sum(self.w2 ** 2))
            return data_loss + l2_loss

        if self.loss == 'mse':
            return np.mean((y_pred-y) ** 2)

    def train(self, X, y, y_pred, epochs, learning_rate):
        for i in range(epochs):
            self.forward(X)
            self.backward(X, y, y_pred)

            self.update_parameters(learning_rate)

    def batch_train(self, X, y, epochs, lr, batch_size, l2_lambda=0.0):
        N = X.shape[0]
        for i in range(epochs):
            idx = np.random.permutation(N)
            X, y = X[idx], y[idx]

            for start in range(0, N, batch_size):
                end = start + batch_size
                Xb, yb = X[start:end], y[start:end]
                y_pred = self.forward(Xb)
                self.backward(Xb, yb, y_pred, l2_lambda)
                self.update_parameters(lr)

            y_pred = self.forward(X)
            loss = self.calculate_loss(y_pred, y, l2_lambda)
            acc = np.mean(np.argmax(y, axis=1) == np.argmax(y_pred, axis=1))
            print(f"epoch {i + 1}/{epochs}  loss: {loss:.4f}  acc: {acc:.4f}")

        self.save()

    def copy(self, nn):
        self.w1 = nn.w1
        self.b1 = nn.b1

        self.w2 = nn.w2
        self.b2 = nn.b2

    def save(self, filename="model.npz"):
        np.savez(filename,
                 w1=self.w1, b1=self.b1,
                 w2=self.w2, b2=self.b2)

    def load(self, filename="model.npz"):
        data = np.load(filename)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]


# df = pd.read_csv("mnist_test.csv")
# data = df.values   # converts to a numpy array
#
# nn = NeuralNetwork(784, 6, 10)
# # nn.load('save1.npz')
#
# X = data[:, 1:] / 255.0
# y = data[:, 0]
# y_one_hot = np.eye(10)[y]
#
# X_train = X[:8000, :]
# y_train = y_one_hot[:8000]
#
# X_test = X[8000:, :]
# y_test = y_one_hot[8000:]
#
# nn.batch_train(X_train, y_train, 1000, 0.1, 100, 1e-2)
#
# y_pred = np.argmax(nn.forward(X_test), axis=1)
# y_true = np.argmax(y_test, axis=1)
#
# accuracy = np.mean(y_pred == y_true)
#
# print(accuracy * 100)