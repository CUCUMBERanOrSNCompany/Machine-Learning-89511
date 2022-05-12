# Machine Learning Assignment 3
# DEVELOPED BY CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, DECEMBER, 2020.
# ALL RIGHTS RESERVED.

# DISCLAIMER: To understand this assignment better, I watched a guide found here:
# https://www.youtube.com/watch?v=0Sfxa1mWK-U

# from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import special
import math

# imageGenerator Main function gets a 2D array of pixels (Each row represents a single object)
# and returns matrices and images.
def imageGenerator(arrayToConvert):
    lineOfTemplate = math.sqrt(len(arrayToConvert[0]))
    colsOfTemplate = lineOfTemplate
    matricesToConvert = []
    arrayOfImages = []
    # arrayToConvert *= 255
    for line in arrayToConvert:
        #for word in line:
         #   word = float(word)
        counter = 0
        rawImage = []
        for index in range(int(lineOfTemplate)):
            lineOfRaw = []
            for subIndex in range(int(colsOfTemplate)):
                lineOfRaw.append(int(line[counter]))
                counter += 1
            rawImage = np.append(rawImage, lineOfRaw)
        # print(rawImage)
        # print(".............................................")
        rawImage = np.array(rawImage).reshape(int(lineOfTemplate), int(colsOfTemplate))
        arrayOfImages.append(plt.imshow(rawImage, cmap='gray'))
       #plt.show()
        matricesToConvert.append(rawImage)
    bundle = [matricesToConvert, arrayOfImages]
    return bundle
# ................................... END OF imageGenerator MAIN FUNCTION...............................................

# oneHotEncoder gets an array of labels and one hot encodes it
def oneHotEncoder(arrayOfLabels):
    returnedArray= []
    oneHotArray = (np.array([])).astype(int)
    for current in arrayOfLabels:
        encode = np.zeros(10).astype(int)
        # print(encode)
        encode[int(current) - 1] = 1
        returnedArray.append(encode)
        # print(current)
        #print(encode)
        # oneHotArray = np.append(oneHotArray, encode)

    # return oneHotArray
    return returnedArray

# ................................... END OF oneHotEncoder MAIN FUNCTION...............................................

# weightsAndBiasCalc calculates the weights based on the following randomizing rule: first value is the number
# of neurons in the input layer, second value is the number of neurons in the ending layer.
# bias is always 1.
# This function is part of the initialization stage.
def weightsAndBiasCalc(inputNeurons, hiddenLayerNeurons, numberOfLabels):
    # print(str(inputNeurons) + " " + str(hiddenLayerNeurons) + " " + str(numberOfLabels))
    # weightOne = np.random.uniform(-0.1, 0.1, [hiddenLayerNeurons, inputNeurons])
    # biasOne = np.random.uniform(-0.1, 0.1, [hiddenLayerNeurons, 1])
    # weightTwo = np.random.uniform(-0.1, 0.1, [numberOfLabels, hiddenLayerNeurons])
    # biasTwo = np.random.uniform(-0.1, 0.1, [numberOfLabels, 1])

    weightOne = np.random.randn(hiddenLayerNeurons, inputNeurons) * 0.1
    biasOne = np.random.randn(hiddenLayerNeurons, 1) * 0.1
    weightTwo = np.random.randn(numberOfLabels, hiddenLayerNeurons) * 0.1
    biasTwo = np.random.randn(numberOfLabels, 1) * 0.1
    retunValues = [weightOne, biasOne, weightTwo, biasTwo]
    # print(len(retunValues[0]))
    # print(len(retunValues[1]))
    # print(len(retunValues[2]))
    # print(len(retunValues[3]))

    return retunValues

# ................................... END OF weightsAndBiasCalc MAIN FUNCTION..........................................

# sigmoid calculates Sigmoid.
def sigmoid(z):
    # result = 1/(1 + np.exp(-z))  #Todo: Check if it suppose to be z or (-z)!
    result = scipy.special.expit(z)
    return result

# ............................................. END OF FUNCTION......................................................

# derOfSigmoid calculates the derivation of sigmoid.
def derOfSigmoid(example):
    result = sigmoid(example) * (1 - sigmoid(example))
    return result

# ............................................. END OF FUNCTION......................................................

# softMax calculates softMax.
def softMax(z):
    result = np.exp(z - np.max(z)) / (np.exp(z - np.max(z))).sum()
    # print(np.min(result))
    return result

# ............................................. END OF FUNCTION......................................................

# forwardPropagation is the first stage after the initialization. We are taking examples
# and pass them through the neural network. Using the weights and bias we found
def forwardPropagation(weights, trainInterX):
    # retunValues = [weightOne, biasOne, weightTwo, biasTwo]
    weightsOne = np.array(weights[0])
    biasOne = np.array(weights[1])
    weightsTwo = weights[2]
    biasTwo = weights[3]

    example.shape = (weightsOne.shape[1], 1)
    zOne = np.dot(weightsOne, trainInterX) + biasOne
    hiddenOne = sigmoid(zOne)
    zTwo = np.dot(weightsTwo, hiddenOne) + biasTwo
    hiddenTwo = softMax(zTwo)
    yHat = np.argmax(hiddenTwo)  # More suitable than hiddenTwo for multi class.

    exampleBundle = [example, zOne, hiddenOne, weightsOne, biasOne, zTwo, hiddenTwo, weightsTwo, biasTwo, yHat]
    return exampleBundle

# ................................... END OF forwardPropagation MAIN FUNCTION..........................................

# backwardPropagation function handles the next step in our model, after all we done in forwardPropagation
def backwardPropagation(forwardCache, labels):
    # [0: example, 1: zOne, 2: hiddenOne, 3: weightsOne, 4: biasOne,
    # 5: zTwo, 6: hiddenTwo, 7: weightsTwo, 8: biasTwo, 9: yHat]
    # Reminder: Derivation of softmax is pk-1, when pk is yHat we found previously. ~31" At the video.
    # print(len(forwardCache))

    labels.shape = (forwardCache[6].shape[0], 1)  # This line is essential for successful calculation of derHTwo
    derZTwo = (forwardCache[6] - labels)  # HiddenTwo - label
    derWTwo = np.dot(derZTwo, forwardCache[2].T)  # np.dot(derZTwo, transform(hiddenOne))
    derBiasTwo = derZTwo
    derHOne = np.dot(forwardCache[7].T, derZTwo)  # np.dot(weightsTwo.transform, derZTwo)
    derZOne = derHOne * derOfSigmoid(forwardCache[1])  # derHOne * derOfSigmoid(zOne)
    derWOne = np.dot(derZOne, forwardCache[0].T)  # np.dot(derZOne, example.Transform
    derBiasOne = derZOne
    result = [derZTwo, derWTwo, derBiasTwo, derHOne, derZOne, derWOne, derBiasOne]
    # print("okay1")

    # print("Done.")
    return result

# ................................... END OF backwardPropagation MAIN FUNCTION..........................................

# update calculates the new weights.
def update(gradient, eta, weights):
    # [0. weightOne, 1. biasOne, 2. weightTwo, 3. biasTwo]
    weightOne = weights[0]
    biasOne = weights[1]
    weightTwo = weights[2]
    biasTwo = weights[3]

    # [0. derZTwo, 1. derWTwo, 2. derBiasTwo,
    # 3. derHOne, 4. derZOne, 5. derWOne, 6. derBiasOne]

    # print(str(len(gradients)) + " " + str(len(weights)))

    # print(len(gradient))
    weightOne -= gradient[5] * eta  # gradient[5] is derivation of weight one.
    biasOne -= gradient[6] * eta  # gradient[6] is derivation of bias one.
    weightTwo -= gradient[1] * eta  # gradient[1] is derivation of weight two.
    biasTwo -= gradient[2] * eta  # gradient[2] is derivation of bias two.
    # print("Okay")

    weight = [weightOne, biasOne, weightTwo, biasTwo]
    return weight

# ................................... END OF update MAIN FUNCTION..........................................

# Predict function predicts.
def predict(examples, labels, weights, labelsDec):
    # [0: example, 1: zOne, 2: hiddenOne, 3: weightsOne, 4: biasOne,
    # 5: zTwo, 6: hiddenTwo, 7: weightsTwo, 8: biasTwo, 9: yHat]

    attempts = 0
    hits = 0
    counter = 0
    # results = forwardPropagation(weights, examples)
    # print(len(labelsDec))

    for result in examples:
        attempts += 1
        result = forwardPropagation(weights, examples[counter])
        # print(np.argmax(result[6]))
        # print(len(examples[0]))
        # print(labels[counter])
        # print(labelsDecimal[counter])

        # print("result[6] " + str(np.argmax(result[6])))
        # print("result[9] " + str(np.argmax(result[9])))
        # print("y " + str(labelsDec[counter]))

        if np.argmax(result[6]) == 9:
            factor = 0
        else:
            factor = np.argmax(result[6]) + 1
        if factor == labelsDec[counter]:
            hits += 1
        # print(np.argmax(result[6]) + 1)

        counter += 1

    accuracy = hits / attempts
    # print("Accuracy: " + str(accuracy))

# .................................. END OF FUNCTION..............................................................

# fileGenerator generates the file
def fileGenerator(examples, weights):
    counter = 0
    cuPredictionArray = []
    for _ in examples:
        finalResult = forwardPropagation(weights, examples[counter])
        if (np.argmax(finalResult[6]) == 9):
            cuPredictionArray.append(0)
        else:
            cuPrediction = np.argmax(finalResult[6]) + 1  # Applying factor
            cuPredictionArray.append(cuPrediction)
        counter += 1

    file = open("test_y", "w+")
    for iterity in cuPredictionArray:
        file.write(str(iterity) + "\n")
    file.close()

    return


debugLimit = 500  # To limit the number of examples during the building
trainX = np.loadtxt("train_x") / 255
trainY = np.loadtxt("train_y")
# trainX = np.loadtxt("train_x") / 255
# trainY = np.loadtxt("train_y")
testX = np.loadtxt("test_x") / 255
mergedExampleSet = []

# imageGenerator(trainX)
oneHotExampleY = oneHotEncoder(trainY)

counter = 0

for example in trainX:
    production = [trainX[counter], oneHotExampleY[counter], trainY[counter]]
    mergedExampleSet.append(production)
    counter += 1

hiddenNeurons = 77

weights = weightsAndBiasCalc(len(trainX[0]), hiddenNeurons, len(oneHotExampleY[0]))

eta = 0.01
epoch = 100

for _ in range(epoch):
    np.random.shuffle(mergedExampleSet)  # Shuffling the examples TOGETHER with the labels.

    theExamples = []
    theLabels = []  # The labels in One Hot Encoding
    labelsDecimal = []  # The labels in regular representation, as provided.
    for merge in mergedExampleSet:  # Splitting the examples from the labels without messing up the connection.
        theExamples.append(merge[0])
        theLabels.append(merge[1])
        labelsDecimal.append(merge[2])

    counter = 0
    for example in theExamples:
        forward = forwardPropagation(weights, example)  # Does the forward phase example by example.
        backwardGrads = backwardPropagation(forward, theLabels[counter])
        # [0. derZTwo, 1. derWTwo, 2. derBiasTwo, 3. derHOne, 4. derZOne, 5. derWOne, 6. derBiasOne]
        # print(len(backwardGrads))
        weights = update(backwardGrads, eta, weights)
        counter += 1

    # for test in theTests:
        # check = forwardPropagation(weights, test)
        # print(np.argmax(check[6]))

    predict(theExamples, theLabels, weights, labelsDecimal)

file = open("test_y", "w+")
for example in testX:
    # production = [testX[counter]]
    result = forwardPropagation(weights, example)
    argMax = np.argmax(result[6])

    if argMax == 9:  # Applying factor
        argMax = 0
    else:
        argMax += 1

    file.write(str(argMax) + "\n")
    # print("Boom: " + str(np.argmax(result[6])))

file.close()
