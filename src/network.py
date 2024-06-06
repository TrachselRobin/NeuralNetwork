import random
from calculations import *


class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_layers, num_neurons_per_hidden_layer, num_outputs):
        self._layers = []  # example [[input, input, etc.], [neuron, neuron, neuron, etc.], [neuron, neuron, neuron, etc.], [output, output, etc.]]
        self._learningRate = 0.1
        self._num_inputs = num_inputs
        self._num_hidden_layers = num_hidden_layers
        self._num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self._num_outputs = num_outputs

        inputLayer = [Neuron(0) for _ in range(num_inputs)]  # Initialize input layer with num_inputs neurons
        self._layers.append(inputLayer)

        for i in range(num_hidden_layers):
            if i == 0:  # first hidden layer so it needs to take amount of inputs
                hiddenLayer = [Neuron(num_inputs) for _ in range(num_neurons_per_hidden_layer)]
            else:
                hiddenLayer = [Neuron(num_neurons_per_hidden_layer) for _ in range(num_neurons_per_hidden_layer)]
            self._layers.append(hiddenLayer)

        outputLayer = [Neuron(num_neurons_per_hidden_layer) for _ in range(num_outputs)]
        self._layers.append(outputLayer)

    def train(self, inputs, expected):
        for i in range(self.num_inputs):
            self._layers[0][i].value = inputs[i]

        for i in range(1, len(self._layers)):
            for neuron in self._layers[i]:
                neuron.value = 0
                for j in range(len(self._layers[i - 1])):
                    neuron.value += self._layers[i - 1][j].value * neuron.weights[j]
                neuron.value = sigmoid(neuron.value)

        for i in range(len(self._layers[-1])):
            self._layers[-1][i].error = expected[i] - self._layers[-1][i].value
            for j in range(len(self._layers[-2])):
                self._layers[-1][i].weights[j] += self.learningRate * self._layers[-1][i].error * self._layers[-2][j].value

        for i in range(len(self._layers) - 2, 0, -1):
            for j in range(len(self._layers[i])):
                error = 0
                for k in range(len(self._layers[i + 1])):
                    error += self._layers[i + 1][k].error * self._layers[i + 1][k].weights[j]
                self._layers[i][j].error = error * sigmoid_derivative(self._layers[i][j].value)
                for k in range(len(self._layers[i][j].weights)):
                    inputValue = inputs[k] if i == 1 else self._layers[i - 1][k].value
                    self._layers[i][j].weights[k] += self.learningRate * self._layers[i][j].error * inputValue

    def guess(self, inputs):
        for i in range(self.num_inputs):
            self._layers[0][i].value = inputs[i]

        for i in range(1, len(self._layers)):
            for neuron in self._layers[i]:
                neuron.value = 0
                for j in range(len(self._layers[i - 1])):
                    neuron.value += self._layers[i - 1][j].value * neuron.weights[j]
                neuron.value = sigmoid(neuron.value)
        return [neuron.value for neuron in self._layers[-1]]

    def save(self, filename):
        pass

    def load(self, filename):
        pass

    def __str__(self):
        pass

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_hidden_layers(self):
        return self._num_hidden_layers

    @property
    def num_neurons_per_hidden_layer(self):
        return self._num_neurons_per_hidden_layer

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def layers(self):
        return self._layers

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, value):
        self._learningRate = value


class Neuron:
    def __init__(self, amountWeights):
        self._weights = [random.uniform(0, 1) for _ in range(amountWeights)]
        self._value = 0
        self._error = 0

    @property
    def weights(self):
        return self._weights

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    @property
    def error(self):
        return self._error

    @error.setter
    def error(self, value):
        self._error = value

    def __str__(self):
        pass