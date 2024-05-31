from generateInput import getRandomInput
from calculations import *


if __name__ == '__main__':
    result = getRandomInput()
    print(f"Calculate: {result['calculation']} \nResult: {result['answer']}")

    for i in range(100):
        print(f"{i}: {i * sigmoid(10)}")
