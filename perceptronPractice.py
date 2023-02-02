import numpy as np
import pandas as pd

def neuron(inputs):
    inputs = pd.Series(inputs)
    return sign((weights * inputs).sum())

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

def f(x):
    # y = mx + b
    return (0.89 * x) - 0.2

def trainingPoint():
    x = (np.random.rand() * 2) - 1
    y = (np.random.rand() * 2) - 1
    label = 0
    bias = 1

    lineY = f(x)
    if y > lineY:
        label = 1
    else:
        label = -1

    return [x, y, bias, label]

def getTrainingData(num):
    training_data = []
    for i in range(0, num):
        training_data.append(trainingPoint())
    return training_data


def train(inputs, target):
    error = target - neuron(inputs)
    for i in range(len(weights)):
        weights.iloc[i] += error * inputs[i] * learning_rate

def trainPerceptron(trainingData):
    correct_sum = 0
    wrong_sum = 0
    for dataPoint in trainingData:
        points = [dataPoint[0], dataPoint[1], dataPoint[2]]
        target = dataPoint[3]

        train(points, target)

        guess = neuron(points)
        if guess == target:
            correct_sum += 1
        else:
            wrong_sum += 1

    print(f'Correct Guesses: {correct_sum}')
    print(f'Wrong Guesses: {wrong_sum}')

def weightInit(n):
    weights = []
    for i in range(0,n):
        weights.append((np.random.rand() * 2) -1)
    return pd.Series(weights)

#####
#MAIN
#####

#INITIALIZE WEIGHTS

weights = weightInit(3)

#GET TRAINING DATA

training_data = getTrainingData(100)

#LEARNING RATE

learning_rate = 0.1

#TRAINING

trainPerceptron(training_data)