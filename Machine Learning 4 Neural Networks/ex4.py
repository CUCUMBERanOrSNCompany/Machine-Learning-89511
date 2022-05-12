# Machine Learning Assignment 4
# DEVELOPED BY Tomer Himi & CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, DECEMBER, 2020.
# ALL RIGHTS RESERVED.

# from PIL import Image
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import special
import math
import torch
from torch import nn
import torch.nn.functional as F


def weightAndBias(neuronNumberArray):
    weightsArray = []
    biasArray = []

    for index in range(len(neuronNumberArray)):
        if index == 0:
            continue
        weight = np.random.randn(neuronNumberArray[index], neuronNumberArray[index - 1]) * 0.1
        bias = np.random.randn(neuronNumberArray[index], 1) * 0.1
        weightsArray.append(weight)
        biasArray.append(bias)
        # print("Weight: " + str(weight) + " Bias: " + str(bias))
    results = [weightsArray, biasArray]
    return results

def linearGenerator(numbOfNeuronsInLayerArr):
    results = []
    index = 0
    for index in range(len(numbOfNeuronsInLayerArr)):
        if index == 0:
            continue
        fullyConnected = nn.Linear(numbOfNeuronsInLayerArr[index - 1], numbOfNeuronsInLayerArr[index])
        results.append(fullyConnected)
    return results

# For chapters 1 to 4. TWO LAYERS REQUIRED (100,50). For chapters 5,6. We need to use FIVE layers (128,64,10,10,10)
def forward(numOfNeuInLayerArr, weightsArray, biasArray, activationFunction):
    npArrayOfWeights = []
    npArrayOfBias = []
    arrayOfResults = []
    fullyConnectedArray = linearGenerator(numOfNeuInLayerArr)

    for index in range(len(weightsArray)):
        weight = np.array(weightsArray[index])
        bias = np.array(biasArray[index])
        npArrayOfWeights.append(weight)
        npArrayOfBias.append(bias)



    for index in range(len(numOfNeuInLayerArr)):
        if index == 0:
            continue
        else:
            # fullyConnected = nn.Linear(numOfNeuInLayerArr[index - 1], numOfNeuInLayerArr[index])
            # fullyConnectedArray.append(fullyConnected)
            # fullyConnected = torch.Tensor(fullyConnected)
            if activationFunction == "relu":
                # print("Entered relu")
                x = relU(fullyConnectedArray[index - 1].weight.data)
                # print("With RelU: " + str(x))
            elif activationFunction == "sigmoid":
                # print("Entered sigmoid")
                x = sigmoid(fullyConnectedArray[index - 1].weight.data)
                # print("With Sigmoid: " + str(x))
            x = F.log_softmax(x, dim=1)  # todo: dim MUST be mentioned EXPLICITLY and could be either -2, -1, 0 OR 1. In an event of a bug, try to change the value of dim.
            # print(x)
            arrayOfResults.append(x)
            # x = F.log_softmax(x)
            # print("Okay")
            # arrayOfResults.append(x)

    return arrayOfResults


    '''
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
    '''

# Regularization - Please refer to lecture 9
def dropout():
    return

# Normalization
def batchNormalization():
    return

# Activation
def relU(valueToCalc):
    result = F.relu(valueToCalc)
    # print(result)
    return result

# Activation
def sigmoid(valueToCalc):
    # print(valueToCalc)
    # print(type(valueToCalc))
    # result = scipy.special.expit(valueToCalc)
    result = torch.sigmoid(valueToCalc)
    # result = 1 / (1 + np.exp(valueToCalc))
    # print(result)
    return result

# For chapters 5,6. We need to use FIVE layers (128,64,10,10,10),
# as OPPOSED to TWO layers (100, 50) in the previous FOUR chapters.
def advancedModelForward():
    return

# Built in function in pyTorch. SoftMax range is [0,1],
# while log-SoftMax range is [0, infty) (Reminder: log of 0 is infty).
def logSoftMax():
    return

def modelGenerator(trainSet, labelsSet, learningRate,
                   numberOfLayers, activationFunction,
                   optimizerOpt, normalization):
    fullyConnectedArr = linearGenerator(numberOfLayers)
    fullyConnectedArrWeightData = []
    for index in range(len(fullyConnectedArr)):
        fullyConnectedArrWeightData.append(fullyConnectedArr[index].weight.data)

    if(optimizerOpt == "SGD"):
        optimizer = torch.optim.SGD(fullyConnectedArrWeightData, lr=learningRate)
    elif(optimizerOpt == "Adam"):
        optimizer = torch.optim.Adam(fullyConnectedArrWeightData, lr=learningRate)

    index = 0
    weightsAndBiasIni = weightAndBias(numberOfLayers)
    for _ in trainSet:  # Doing the process for EACH example and correspond label INDIVIDUALLY
        optimizer.zero_grad()
        forwardStage = forward(numberOfLayers, weightsAndBiasIni[0], weightsAndBiasIni[1], activationFunction)
        loss = F.nll_loss(forwardStage[index], labelsSet[index])
        loss.backward()
        optimizer.step()
        index += 1
    return

# .................................................. MAIN .............................................................

debugLimit = 500  # To limit the number of examples during the building
trainX = np.loadtxt("train_x", max_rows=debugLimit) / 255
trainY = np.loadtxt("train_y", max_rows=debugLimit)

check = [1, 2, 3, 4]  # Number of neurons in each layer
weightsAndBias = weightAndBias(check)
learningRate = 0.01
# numOfNeuInLayerArr, weightsArray, biasArray, activationFunction

# test = forward(check, weightsAndBias[0], weightsAndBias[1], "relu")
test = modelGenerator(trainX, trainY, learningRate, check, "relu", "SGD", "none")
print("Hi")

# backward(test)

# trainX = np.loadtxt(sys.argv[1]) / 255  # train_x
# trainY = np.loadtxt(sys.argv[2])  # train_y
# testX = np.loadtxt(sys.argv[3]) / 255  # test_x
#  test_y is not an argument! Produce it in the end.

epoch = 10
# PLEASE DIVIDE THE TRAINING SET TO 80 TRAINING AND 20 VALIDATION

class FirstNet(nn.Module):
    def __init__(self, image_size):
        super(FirstNet, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)
