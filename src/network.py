import random


class NeuralNetwork:
    def __init__(self, amountInputNeurons, amountHiddenNeurons, amountHiddenLayers, amountOutputNeurons):
        self.amountInputNeurons = amountInputNeurons
        self.amountHiddenNeurons = amountHiddenNeurons
        self.amountHiddenLayers = amountHiddenLayers
        self.amountOutputNeurons = amountOutputNeurons
        self.inputLayer = []
        self.hiddenLayers = []
        self.outputLayer = []
        self.learningRate = 0.1

    def newModel(self):
        self.inputLayer = [InputNeuron(self) for i in range(self.amountInputNeurons)]
        self.hiddenLayers = [HiddenLayer(self.amountHiddenNeurons) for i in range(self.amountHiddenLayers)]
        self.outputLayer = [HiddenNeuron() for i in range(self.amountOutputNeurons)]

    def model(self, file):
        with open(file, "r") as f:
            lines = f.readlines()
            self.amountInputNeurons = int(lines[0])
            self.amountHiddenNeurons = int(lines[1])
            self.amountHiddenLayers = int(lines[2])
            self.amountOutputNeurons = int(lines[3])
            self.learningRate = float(lines[4])
            self.inputLayer = [InputNeuron(self) for i in range(self.amountInputNeurons)]
            self.hiddenLayers = [HiddenLayer(self.amountHiddenNeurons) for i in range(self.amountHiddenLayers)]
            self.outputLayer = [HiddenNeuron() for i in range(self.amountOutputNeurons)]
            i = 5
            for neuron in self.inputLayer:
                neuron.weights = [float(x) for x in lines[i].split()]
                i += 1
            for layer in self.hiddenLayers:
                for neuron in layer.neurons:
                    neuron.weights = [float(x) for x in lines[i].split()]
                    i += 1
            for neuron in self.outputLayer:
                neuron.weights = [float(x) for x in lines[i].split()]
                i += 1


class InputNeuron:
    def __init__(self, network):
        self.value = 0
        self.weights = []
        self.bias = 0

        for i in range(network.amountInputNeurons):
            self.weights.append(random.random())


class HiddenNeuron:
    def __init__(self):
        self.value = 0
        self.weights = []
        self.bias = 0


class HiddenLayer:
    def __init__(self, amountNeurons):
        self.neurons = [HiddenNeuron() for i in range(amountNeurons)]
