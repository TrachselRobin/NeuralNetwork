from calculations import *
from random import random


class NeuralNetwork:
    def __init__(self, amountInputNeurons, amountHiddenLayers, amountHiddenNeurons, amountOutputNeurons,
                 learningRate=1):
        self._inputNeurons = [InputNeuron() for _ in range(amountInputNeurons)]
        self._hiddenLayers = [HiddenLayer(amountHiddenNeurons, i, amountInputNeurons if i == 0 else amountHiddenNeurons)
                              for i in range(amountHiddenLayers)]
        self._outputNeurons = [OutputNeuron(amountHiddenNeurons) for _ in range(amountOutputNeurons)]
        self._learningRate = learningRate

    def guess(self, inputs: []):
        for i in range(self.amountInputNeurons):
            self.inputNeurons[i].value = inputs[i]

        for hiddenLayer in self.hiddenLayers:
            for hiddenNeuron in hiddenLayer.hiddenNeurons:
                hiddenNeuron.value = 0
                for j, inputNeuron in enumerate(
                        self.inputNeurons if hiddenLayer == self.hiddenLayers[0] else self.hiddenLayers[
                            self.hiddenLayers.index(hiddenLayer) - 1].hiddenNeurons):
                    hiddenNeuron.value += inputNeuron.value * hiddenNeuron.weights[j]
                hiddenNeuron.value = sigmoid(hiddenNeuron.value)

        for outputNeuron in self.outputNeurons:
            outputNeuron.value = 0
            for hiddenNeuron in self.hiddenLayers[-1].hiddenNeurons:
                outputNeuron.value += hiddenNeuron.value * outputNeuron.weights[
                    self.hiddenLayers[-1].hiddenNeurons.index(hiddenNeuron)]
            # No sigmoid activation here
        return [outputNeuron.value for outputNeuron in self.outputNeurons]

    def train(self, inputs: [], expected: []):
        self.guess(inputs)

        # Output neuron error and weight update
        for i, outputNeuron in enumerate(self.outputNeurons):
            outputNeuron.error = expected[i] - outputNeuron.value
            for j, hiddenNeuron in enumerate(self.hiddenLayers[-1].hiddenNeurons):
                outputNeuron.weights[j] += self.learningRate * outputNeuron.error * hiddenNeuron.value

        # Hidden neuron error and weight update
        for i in range(self.amountHiddenLayers - 1):
            hiddenLayer = self.hiddenLayers[i]
            nextLayer = self.outputNeurons if i == self.amountHiddenLayers - 1 else self.hiddenLayers[
                i + 1].hiddenNeurons
            for j, hiddenNeuron in enumerate(hiddenLayer.hiddenNeurons):
                error = 0
                if i == self.amountHiddenLayers - 1:
                    for outputNeuron in nextLayer:
                        error += outputNeuron.error * outputNeuron.weights[j]
                else:
                    for nextNeuron in nextLayer:
                        error += nextNeuron.error * nextNeuron.weights[j]
                hiddenNeuron.error = error
                for k in range(hiddenNeuron.amountWeights):
                    inputValue = inputs[k] if i == 0 else self.hiddenLayers[i - 1].hiddenNeurons[k].value
                    hiddenNeuron.weights[k] += self.learningRate * hiddenNeuron.error * sigmoid_derivative(
                        hiddenNeuron.value) * inputValue

    @property
    def amountInputNeurons(self):
        return len(self._inputNeurons)

    @property
    def amountHiddenLayers(self):
        return len(self._hiddenLayers)

    @property
    def amountHiddenNeurons(self):
        return self._hiddenLayers[0].amountNeurons

    @property
    def amountOutputNeurons(self):
        return len(self._outputNeurons)

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, value):
        self._learningRate = value

    @property
    def inputNeurons(self):
        return self._inputNeurons

    @property
    def hiddenLayers(self):
        return self._hiddenLayers

    @property
    def outputNeurons(self):
        return self._outputNeurons


class InputNeuron:
    def __init__(self):
        self._value = 0

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value


class HiddenLayer:
    def __init__(self, amountNeurons, index, amountInputNeurons):
        self._hiddenNeurons = [HiddenNeuron(amountInputNeurons if index == 0 else amountNeurons) for _ in
                               range(amountNeurons)]

    @property
    def amountNeurons(self):
        return len(self._hiddenNeurons)

    @property
    def hiddenNeurons(self):
        return self._hiddenNeurons


class HiddenNeuron:
    def __init__(self, amountWeights):
        self._weights = [random() for _ in range(amountWeights)]
        self._value = 0
        self._error = 0

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

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

    @property
    def amountWeights(self):
        return len(self._weights)


class OutputNeuron:
    def __init__(self, num_weights):
        self._weights = [random() for _ in range(num_weights)]
        self._value = 0
        self._error = 0

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

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

    @property
    def amountWeights(self):
        return len(self._weights)
