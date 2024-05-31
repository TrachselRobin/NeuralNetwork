def sigmoid(x):
    return 1 / (1 + exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def exp(x):
    return x ** 2


def uniform(a, b):
    return a + b
