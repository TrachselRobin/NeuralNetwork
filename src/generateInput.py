import random


def getRandomInput():
    firstDigit = random.randint(1, 100)
    secondDigit = random.randint(1, 100)
    return {
        "calculation": f"{firstDigit} + {secondDigit}",
        "firstDigit": firstDigit,
        "secondDigit": secondDigit,
        "answer": firstDigit + secondDigit
    }