import numpy as np

def load(filename):
    with open(filename, "r") as src:
        lines = [line.strip().split() for line in src]
    x = np.array([line[:-1] for line in lines], dtype=np.float32)
    y = np.array([line[-1] for line in lines], dtype=np.int32)
    return x, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(object):
    def __init__(self, dim):
        self.weight = np.zeros(dim)
        self.bias = 0

    def logit(self, x): # x.shape == (data_size, dim)
        return np.dot(x, self.weight) + self.bias # shape == (data_size,)

    def prob(self, x):
        return sigmoid(self.logit(x))  # shape == (data_size,)

    def loss(self, x, y): # x.shape == (data_size, dim), y.shape == (data_size,)
        logit = self.logit(x)
        losses = np.log(1 + np.exp(logit)) - y * logit # losses.shape == (data_size,)
        return np.mean(losses)

    def loss_grad(self, x, y):
        grad_bs = self.prob(x) - y
        grad_b = np.sum(grad_bs)
        grad_w = np.dot(grad_bs, x)
        return grad_w, grad_b

    def train(self, x, y, iters, lr=1e-8):
        for n in range(iters):
            print("Iteration:", n, "Loss:", self.loss(x, y))
            grad_w, grad_b = self.loss_grad(x, y)
            self.weight = self.weight - lr * grad_w
            self.bias = self.bias - lr * grad_b

    def predict(self, x):
        return np.rint(self.prob(x))

    def evaluate(self, x, y):
        result = self.predict(x)
        print("Accuracy:", np.sum(result == y) / x.shape[0])

if __name__ == "__main__":
    x_train, y_train = load("data/train.txt")
    x_test, y_test = load("data/test.txt")

    classifier = LogisticRegression(x_train.shape[1])
    for i in range(20):
        classifier.train(x_train, y_train, iters=2, lr=1e-9)
        classifier.evaluate(x_test, y_test)


