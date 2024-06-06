from neuralNetwork import NeuralNetwork
from random import randint


def generate_training_data(num_samples):
    training_data = []
    for _ in range(num_samples):
        a = randint(1, 100)
        b = randint(1, 100)
        training_data.append(([a, b], [a + b]))
    return training_data


def train(network, training_data, epochs=1000):
    total_steps = epochs * len(training_data)
    step = 0
    for epoch in range(epochs):
        for inputs, expected in training_data:
            network.train(inputs, expected)
            step += 1
            if step % (total_steps // 100) == 0:
                progress = (step / total_steps) * 100
                print(f'Training progress: {progress:.2f}%')
        if epoch % 100 == 0:
            print(f'Epoch {epoch} complete')


def networkTesting(network):
    test_cases = [
        (2, 13),
        (50, 50),
        (25, 75),
        (99, 1),
        (47, 53)
    ]
    for a, b in test_cases:
        result = network.guess([a, b])[0]
        print(f"Guessing sum of {a} and {b}: {result:.4f} (Expected: {a + b})")


def train_until_correct(network, firstNumber, secondNumber, target_sum=18, tolerance=0.01, max_iterations=10000):
    inputs = [firstNumber, secondNumber]
    expected = [target_sum]

    for iteration in range(max_iterations):
        network.train(inputs, expected)
        output = network.guess(inputs)[0]
        if abs(output - target_sum) < tolerance:
            print(f"Training complete after {iteration + 1} iterations.")
            break
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: Current output = {output:.4f}")
    else:
        print("Max iterations reached without achieving the desired accuracy.")


def guessTesting(network):
    inputs = [5, 13]
    result = network.guess(inputs)[0]
    print(f"Guessing sum of {inputs[0]} and {inputs[1]}: {result:.4f} (Expected: 18)")
    inputs = [1, 1]
    result = network.guess(inputs)[0]
    print(f"Guessing sum of {inputs[0]} and {inputs[1]}: {result:.4f} (Expected: 18)")


def whileTrain(network):
    # train two random digits from 1 to 100, but only print the accuracy in percentage
    iterations = 0
    highPercentage = 0
    while True:
        firstDigit = randint(1, 100)
        secondDigit = randint(1, 100)
        network.train([firstDigit, secondDigit], [firstDigit + secondDigit])
        guess = network.guess([firstDigit, secondDigit])
        accuracy = calculate_accuracy(firstDigit + secondDigit, guess)
        if accuracy >= 99.9:
            highPercentage += 1
            print(f"Accuracy: {accuracy}%")
            print(f"First digit: {firstDigit}\nSecond digit: {secondDigit}\nResult: {firstDigit + secondDigit}\nGuess: {guess[0]}\n{highPercentage}/{iterations}\n")

        iterations += 1


def calculate_accuracy(expected, actual):
    actual_value = actual[0]  # Extract the actual value from the list
    if expected == actual_value:
        return 100.0
    else:
        error_ratio = abs(actual_value - expected) / expected
        accuracy = max(0, 100 - error_ratio * 100)
        return accuracy


if __name__ == '__main__':
    # Initialize the network
    network = NeuralNetwork(amountInputNeurons=2, amountHiddenLayers=1, amountHiddenNeurons=100, amountOutputNeurons=199)

    # Generate training data
    # training_data = generate_training_data(10000)

    # Train the network
    # train(network, training_data)

    # Test the network
    # networkTesting(network)

    # Alternatively, train until a specific sum is correctly guessed
    # train_until_correct(network, 1, 99)
    # train_until_correct(network, 5, 13)

    # Test the network
    # guessTesting(network)

    # train until stoped
    whileTrain(network)
