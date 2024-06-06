from network import NeuralNetwork
from random import randint


def whileTrain(network):
    # train two random digits from 1 to 100, but only print the accuracy in percentage
    iterations = 0
    highPercentage = 0
    while True:
        firstDigit = randint(0, 100)
        secondDigit = randint(0, 100)
        network.train([firstDigit, secondDigit], trainingOutput(firstDigit + secondDigit))
        guess = network.guess([firstDigit, secondDigit])
        result = guess.index(max(guess))
        accuracy = calculate_accuracy(firstDigit + secondDigit, guess)
        if accuracy >= 50:
            highPercentage += 1
            print(f"Accuracy: {accuracy}%")
            print(
                f"First digit: {firstDigit}\nSecond digit: {secondDigit}\nResult: {firstDigit + secondDigit}\nGuess: {result}\n{highPercentage}/{iterations}\n")

        iterations += 1


def calculate_accuracy(expected, actual):
    actual_value = actual.index(max(actual))
    if expected == actual_value:
        return 100.0
    else:
        if expected == 0:
            error_ratio = abs(actual_value - expected)
        else:
            error_ratio = abs(actual_value - expected) / expected
        accuracy = max(0, 100 - error_ratio * 100)
        return accuracy


def trainingOutput(sum_result):
    output = [0] * 201
    output[sum_result] = 1
    return output


if __name__ == '__main__':
    # Initialize the network
    network = NeuralNetwork(num_inputs=2, num_hidden_layers=2, num_neurons_per_hidden_layer=50, num_outputs=201)

    # guess 51 + 35
    inputs = [51, 35]
    result = network.guess(inputs)
    # get the highest activation index
    result = result.index(max(result))
    print(f"1. Guessing sum of {inputs[0]} and {inputs[1]}: {result} (Expected: 86)")

    # train until 99.9% accuracy
    whileTrain(network)

    # guess 5 + 13
    inputs = [5, 13]
    result = network.guess(inputs)
    print(f"2. Guessing sum of {inputs[0]} and {inputs[1]}: {result} (Expected: 18)")
