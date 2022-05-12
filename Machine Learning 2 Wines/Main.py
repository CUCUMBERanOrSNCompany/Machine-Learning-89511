# Machine Learning Assignment 2
# DEVELOPED BY CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, NOVEMBER, 2020.
# ALL RIGHTS RESERVED.

import sys
import math
import numpy as np


# A class that defines a wine object, using the given 12 parameters

class Wine:
    def __init__(self, fixed, volatile, citric,
                 residual, chlorides, free, total,
                 density, ph, sulphates, alcohol, white, red):
        self.fixed = fixed
        self.volatile = volatile
        self.citric = citric
        self.residual = residual
        self.chlorides = chlorides
        self.free = free
        self.total = total
        self.density = density
        self.ph = ph
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.white = white
        self.red = red

    def debugMethod(self):  # A method that shows values of a wine for debug purposes.
        print("[0]: " + str(self.fixed) + " [1]: " + str(self.volatile) + " [2]: " + str(self.citric) +
              " [3]: " + str(self.residual) + " [4]: " + str(self.chlorides) + " [5]: " + str(self.free) +
              " [6]: " + str(self.total) + " [7]: " + str(self.density) + " [8]: " + str(self.ph) +
              " [9]: " + str(self.sulphates) + " [10]: " + str(self.alcohol) +
              " [White]: " + str(self.white) + " [Red]: " + str(self.red))

# Winery is a class that holds and manage Wine objects

class Winery:
    def __init__(self):
        self.catalog = []

    def addToCatalog(self, wine):
        self.catalog.append(wine)

# WineTester class receives a catalog of Winery and create objects that represent normalized values, minimum and maximum values.

class WineTester:
    def __init__(self, wineryCatalog):
        self.wineryCatalog = wineryCatalog
        self.minimumValues = Wine
        self.maximumValues = Wine
        self.averageValues = Wine
        self.stdDevValues = Wine
        self.initializeValues()  # To initialize essential values.

    # A method that initialize min,max, average and standard deviation.
    def initializeValues(self):
        self.findMinimum()
        self.findMaximum()
        self.findAverage()
        self.findStdDeviation()

    def valuesDivider(self, wineToDivide, divider):
        # print(float(divider))
        # wineToDivide.debugMethod()
        wineToDivide.fixed /= divider
        wineToDivide.volatile /= divider
        wineToDivide.citric /= divider
        wineToDivide.residual /= divider
        wineToDivide.chlorides /= divider
        wineToDivide.free /= divider
        wineToDivide.total /= divider
        wineToDivide.density /= divider
        wineToDivide.ph /= divider
        wineToDivide.sulphates /= divider
        wineToDivide.alcohol /= divider
        wineToDivide.white /= divider
        wineToDivide.red /= divider
        return wineToDivide

    def valuesSqrt(self, wineToSqrt):
        # wineToSqrt.debugMethod()
        wineToSqrt.fixed = math.sqrt(wineToSqrt.fixed)
        wineToSqrt.volatile = math.sqrt(wineToSqrt.volatile)
        wineToSqrt.citric = math.sqrt(wineToSqrt.citric)
        wineToSqrt.residual = math.sqrt(wineToSqrt.residual)
        wineToSqrt.chlorides = math.sqrt(wineToSqrt.chlorides)
        wineToSqrt.free = math.sqrt(wineToSqrt.free)
        wineToSqrt.total = math.sqrt(wineToSqrt.total)
        wineToSqrt.density = math.sqrt(wineToSqrt.density)
        wineToSqrt.ph = math.sqrt(wineToSqrt.ph)
        wineToSqrt.sulphates = math.sqrt(wineToSqrt.sulphates)
        wineToSqrt.alcohol = math.sqrt(wineToSqrt.alcohol)
        wineToSqrt.white = math.sqrt(wineToSqrt.white)
        wineToSqrt.red = math.sqrt(wineToSqrt.red)
        return wineToSqrt

    def findMinimum(self):
        if (len(self.wineryCatalog.catalog) > 0):
            minFixed = self.wineryCatalog.catalog[0].fixed
            minVolatile = self.wineryCatalog.catalog[0].volatile
            minCitric = self.wineryCatalog.catalog[0].citric
            minResidual = self.wineryCatalog.catalog[0].residual
            minChlorides = self.wineryCatalog.catalog[0].chlorides
            minFree = self.wineryCatalog.catalog[0].free
            minTotal = self.wineryCatalog.catalog[0].total
            minDensity = self.wineryCatalog.catalog[0].density
            minPh = self.wineryCatalog.catalog[0].ph
            minSulphates = self.wineryCatalog.catalog[0].sulphates
            minAlcohol = self.wineryCatalog.catalog[0].alcohol
            minWhite = self.wineryCatalog.catalog[0].white
            minRed = self.wineryCatalog.catalog[0].red

            for wine in self.wineryCatalog.catalog:
                if (float(wine.fixed) < float(minFixed)):
                    minFixed = wine.fixed

                if (float(wine.volatile) < float(minVolatile)):
                    minVolatile = wine.volatile

                if (float(wine.citric) < float(minCitric)):
                    minCitric = wine.citric

                if (float(wine.residual) < float(minResidual)):
                    minResidual = wine.residual

                if (float(wine.chlorides) < float(minChlorides)):
                    minChlorides = wine.chlorides

                if (float(wine.free) < float(minFree)):
                    minFree = wine.free

                if (float(wine.total) < float(minTotal)):
                    minTotal = wine.total

                if (float(wine.density) < float(minDensity)):
                    minDensity = wine.density

                if (float(wine.ph) < float(minPh)):
                    minPh = wine.ph

                if (float(wine.sulphates) < float(minSulphates)):
                    minSulphates = wine.sulphates

                if (float(wine.alcohol) < float(minAlcohol)):
                    minAlcohol = wine.alcohol

                if (float(wine.white) < float(minWhite)):
                    minWhite = wine.white

                if (float(wine.red) < float(minRed)):
                    minRed = wine.red

        # print(str(minFixed))

        self.minimumValues = Wine(minFixed, minVolatile, minCitric, minResidual, minChlorides, minFree, minTotal,
                                  minDensity,
                                  minPh, minSulphates, minAlcohol, minWhite, minRed)

        # self.minimumValues.debugMethod()

    def findMaximum(self):
        if (len(self.wineryCatalog.catalog) > 0):
            maxFixed = self.wineryCatalog.catalog[0].fixed
            maxVolatile = self.wineryCatalog.catalog[0].volatile
            maxCitric = self.wineryCatalog.catalog[0].citric
            maxResidual = self.wineryCatalog.catalog[0].residual
            maxChlorides = self.wineryCatalog.catalog[0].chlorides
            maxFree = self.wineryCatalog.catalog[0].free
            maxTotal = self.wineryCatalog.catalog[0].total
            maxDensity = self.wineryCatalog.catalog[0].density
            maxPh = self.wineryCatalog.catalog[0].ph
            maxSulphates = self.wineryCatalog.catalog[0].sulphates
            maxAlcohol = self.wineryCatalog.catalog[0].alcohol
            maxWhite = self.wineryCatalog.catalog[0].white
            maxRed = self.wineryCatalog.catalog[0].red

            for wine in self.wineryCatalog.catalog:
                if (float(wine.fixed) > float(maxFixed)):
                    maxFixed = wine.fixed

                if (float(wine.volatile) > float(maxVolatile)):
                    maxVolatile = wine.volatile

                if (float(wine.citric) > float(maxCitric)):
                    maxCitric = wine.citric

                if (float(wine.residual) > float(maxResidual)):
                    maxResidual = wine.residual

                if (float(wine.chlorides) > float(maxChlorides)):
                    maxChlorides = wine.chlorides

                if (float(wine.free) > float(maxFree)):
                    maxFree = wine.free

                if (float(wine.total) > float(maxTotal)):
                    maxTotal = wine.total

                if (float(wine.density) > float(maxDensity)):
                    maxDensity = wine.density

                if (float(wine.ph) > float(maxPh)):
                    maxPh = wine.ph

                if (float(wine.sulphates) > float(maxSulphates)):
                    maxSulphates = wine.sulphates

                if (float(wine.alcohol) > float(maxAlcohol)):
                    maxAlcohol = wine.alcohol

                if (float(wine.white) > float(maxWhite)):
                    maxWhite = wine.white

                if (float(wine.red) > float(maxRed)):
                    maxRed = wine.red

        # print(str(minFixed))

        self.maximumValues = Wine(maxFixed, maxVolatile, maxCitric, maxResidual, maxChlorides, maxFree, maxTotal,
                                  maxDensity, maxPh, maxSulphates, maxAlcohol, maxWhite, maxRed)
        # self.maximumValues.debugMethod()

    def findAverage(self):
        if (len(self.wineryCatalog.catalog) > 0):
            avgFixed = 0
            avgVolatile = 0
            avgCitric = 0
            avgResidual = 0
            avgChlorides = 0
            avgFree = 0
            avgTotal = 0
            avgDensity = 0
            avgPh = 0
            avgSulphates = 0
            avgAlcohol = 0
            avgWhite = 0
            avgRed = 0

            for wine in self.wineryCatalog.catalog:
                avgFixed += float(wine.fixed)
                avgVolatile += float(wine.volatile)
                avgCitric += float(wine.citric)
                avgResidual += float(wine.residual)
                avgChlorides += float(wine.chlorides)
                avgFree += float(wine.free)
                avgTotal += float(wine.total)
                avgDensity += float(wine.density)
                avgPh += float(wine.ph)
                avgSulphates += float(wine.sulphates)
                avgAlcohol += float(wine.alcohol)
                avgWhite += float(wine.white)
                avgRed += float(wine.red)

        self.averageValues = Wine(avgFixed, avgVolatile, avgCitric, avgResidual, avgChlorides, avgFree, avgTotal,
                                  avgDensity, avgPh, avgSulphates, avgAlcohol, avgWhite, avgRed)
        # self.averageValues.debugMethod()
        self.averageValues = self.valuesDivider(self.averageValues, len(self.wineryCatalog.catalog))
        # self.averageValues.debugMethod()

    def findStdDeviation(self):
        stdFixed = 0
        stdVolatile = 0
        stdCitric = 0
        stdResidual = 0
        stdChlorides = 0
        stdFree = 0
        stdTotal = 0
        stdDensity = 0
        stdPh = 0
        stdSulphates = 0
        stdAlcohol = 0
        stdWhite = 0
        stdRed = 0

        if (len(self.wineryCatalog.catalog) > 0):
            for wine in self.wineryCatalog.catalog:
                stdFixed += ((float(wine.fixed) - float(self.averageValues.fixed)) ** 2)
                # print(str(stdFixed))
                stdVolatile += ((float(wine.volatile) - float(self.averageValues.volatile)) ** 2)
                stdCitric += ((float(wine.citric) - float(self.averageValues.citric)) ** 2)
                stdResidual += ((float(wine.residual) - float(self.averageValues.residual)) ** 2)
                stdChlorides += ((float(wine.chlorides) - float(self.averageValues.chlorides)) ** 2)
                stdFree += ((float(wine.free) - float(self.averageValues.free)) ** 2)
                stdTotal += ((float(wine.total) - float(self.averageValues.total)) ** 2)
                stdDensity += ((float(wine.density) - float(self.averageValues.density)) ** 2)
                stdPh += ((float(wine.ph) - float(self.averageValues.ph)) ** 2)
                stdSulphates += ((float(wine.sulphates) - float(self.averageValues.sulphates)) ** 2)
                stdAlcohol += ((float(wine.alcohol) - float(self.averageValues.alcohol)) ** 2)
                stdWhite += ((float(wine.white) - float(self.averageValues.white)) ** 2)
                stdRed += ((float(wine.red) - float(self.averageValues.red)) ** 2)

            self.stdDevValues = Wine(stdFixed, stdVolatile, stdCitric, stdResidual, stdChlorides, stdFree, stdTotal,
                                     stdDensity, stdPh, stdSulphates, stdAlcohol, stdWhite, stdRed)
            # self.stdDevValues.debugMethod()
            self.stdDevValues = self.valuesDivider(self.stdDevValues, len(self.wineryCatalog.catalog))
            # self.stdDevValues.debugMethod()
            self.stdDevValues = self.valuesSqrt(self.stdDevValues)
            # self.stdDevValues.debugMethod()

        self.stdDevValues = Wine(stdFixed, stdVolatile, stdCitric, stdResidual, stdChlorides, stdFree, stdTotal,
                                 stdDensity, stdPh, stdSulphates, stdAlcohol, stdWhite, stdRed)

    # https://en.wikipedia.org/wiki/Feature_scaling

    def minMaxNormalization(self,
                            outputWinery):  # A method that normalize all wines using min-max normalization, and add data to a separate database.
        for wine in self.wineryCatalog.catalog:
            fixed = (float(wine.fixed) - float(self.minimumValues.fixed)) / (
                    float(self.maximumValues.fixed) - float(self.minimumValues.fixed))
            volatile = (float(wine.volatile) - float(self.minimumValues.volatile)) / (
                    float(self.maximumValues.volatile) - float(self.minimumValues.volatile))
            citric = (float(wine.citric) - float(self.minimumValues.citric)) / (
                    float(self.maximumValues.citric) - float(self.minimumValues.citric))
            residual = (float(wine.residual) - float(self.minimumValues.residual)) / (
                    float(self.maximumValues.residual) - float(self.minimumValues.residual))
            chlorides = (float(wine.chlorides) - float(self.minimumValues.chlorides)) / (
                    float(self.maximumValues.chlorides) - float(self.minimumValues.chlorides))
            free = (float(wine.free) - float(self.minimumValues.free)) / (
                    float(self.maximumValues.free) - float(self.minimumValues.free))
            total = (float(wine.total) - float(self.minimumValues.total)) / (
                    float(self.maximumValues.total) - float(self.minimumValues.total))
            density = (float(wine.density) - float(self.minimumValues.density)) / (
                    float(self.maximumValues.density) - float(self.minimumValues.density))
            ph = (float(wine.ph) - float(self.minimumValues.ph)) / (
                    float(self.maximumValues.ph) - float(self.minimumValues.ph))
            sulphates = (float(wine.sulphates) - float(self.minimumValues.sulphates)) / (
                    float(self.maximumValues.sulphates) - float(self.minimumValues.sulphates))
            alcohol = (float(wine.alcohol) - float(self.minimumValues.alcohol)) / (
                    float(self.maximumValues.alcohol) - float(self.minimumValues.alcohol))
            white = (float(wine.white) - float(self.minimumValues.white)) / (
                    float(self.maximumValues.white) - float(self.minimumValues.white))
            red = (float(wine.red) - float(self.minimumValues.red)) / (
                    float(self.maximumValues.red) - float(self.minimumValues.red))

            normalWine = Wine(fixed, volatile, citric, residual, chlorides,
                              free, total, density, ph, sulphates, alcohol, white, red)

            outputWinery.addToCatalog(normalWine)
            # normalWine.debugMethod()

    def meanNormalization(self,
                          outputWinery):  # A method that normalize all wines using mean normalization, and add data to a separate database.
        for wine in self.wineryCatalog.catalog:
            fixed = (float(wine.fixed) - float(self.averageValues.fixed)) / (
                    float(self.maximumValues.fixed) - float(self.minimumValues.fixed))
            volatile = (float(wine.volatile) - float(self.averageValues.volatile)) / (
                    float(self.maximumValues.volatile) - float(self.minimumValues.volatile))
            citric = (float(wine.citric) - float(self.averageValues.citric)) / (
                    float(self.maximumValues.citric) - float(self.minimumValues.citric))
            residual = (float(wine.residual) - float(self.averageValues.residual)) / (
                    float(self.maximumValues.residual) - float(self.minimumValues.residual))
            chlorides = (float(wine.chlorides) - float(self.averageValues.chlorides)) / (
                    float(self.maximumValues.chlorides) - float(self.minimumValues.chlorides))
            free = (float(wine.free) - float(self.averageValues.free)) / (
                    float(self.maximumValues.free) - float(self.minimumValues.free))
            total = (float(wine.total) - float(self.averageValues.total)) / (
                    float(self.maximumValues.total) - float(self.minimumValues.total))
            density = (float(wine.density) - float(self.averageValues.density)) / (
                    float(self.maximumValues.density) - float(self.minimumValues.density))
            ph = (float(wine.ph) - float(self.averageValues.ph)) / (
                    float(self.maximumValues.ph) - float(self.minimumValues.ph))
            sulphates = (float(wine.sulphates) - float(self.averageValues.sulphates)) / (
                    float(self.maximumValues.sulphates) - float(self.minimumValues.sulphates))
            alcohol = (float(wine.alcohol) - float(self.averageValues.alcohol)) / (
                    float(self.maximumValues.alcohol) - float(self.minimumValues.alcohol))
            white = (float(wine.white) - float(self.averageValues.white)) / (
                    float(self.maximumValues.white) - float(self.minimumValues.white))
            red = (float(wine.red) - float(self.averageValues.red)) / (
                    float(self.maximumValues.red) - float(self.minimumValues.red))

            normalWine = Wine(fixed, volatile, citric, residual, chlorides,
                              free, total, density, ph, sulphates, alcohol, white, red)

            outputWinery.addToCatalog(normalWine)
            # normalWine.debugMethod()

    def zScoreNormalization(self,
                            outputWinery):  # A method that normalize all wines using mean normalization, and add data to a separate database.
        for wine in self.wineryCatalog.catalog:
            fixed = (float(wine.fixed) - float(self.averageValues.fixed)) / float(self.stdDevValues.fixed)
            volatile = (float(wine.volatile) - float(self.averageValues.volatile)) / float(self.stdDevValues.volatile)
            citric = (float(wine.citric) - float(self.averageValues.citric)) / float(self.stdDevValues.citric)
            residual = (float(wine.residual) - float(self.averageValues.residual)) / float(self.stdDevValues.residual)
            chlorides = (float(wine.chlorides) - float(self.averageValues.chlorides)) / float(
                self.stdDevValues.chlorides)
            free = (float(wine.free) - float(self.averageValues.free)) / float(self.stdDevValues.free)
            total = (float(wine.total) - float(self.averageValues.total)) / float(self.stdDevValues.total)
            density = (float(wine.density) - float(self.averageValues.density)) / float(self.stdDevValues.density)
            ph = (float(wine.ph) - float(self.averageValues.ph)) / float(self.stdDevValues.ph)
            sulphates = (float(wine.sulphates) - float(self.averageValues.sulphates)) / float(
                self.stdDevValues.sulphates)
            alcohol = (float(wine.alcohol) - float(self.averageValues.alcohol)) / float(self.stdDevValues.alcohol)
            white = (float(wine.white) - float(self.averageValues.white)) / float(self.stdDevValues.white)
            red = (float(wine.red) - float(self.averageValues.red)) / float(self.stdDevValues.red)

            normalWine = Wine(fixed, volatile, citric, residual, chlorides,
                              free, total, density, ph, sulphates, alcohol, white, red)

            outputWinery.addToCatalog(normalWine)
            # normalWine.debugMethod()

# ............................MAIN..............................

# wineryCreator is a method that get lines of a text file and assemble a Winery object.
def wineryCreator(linesOfWine):
    # print(len(linesOfWine))
    wineryToCreate = Winery()
    for rawLine in linesOfWine:
        words = rawLine.split(",")
        if (words[11].rstrip() == "W"):  # One Hot Encoding for the type: https://en.wikipedia.org/wiki/One-hot
            oneHotWhite = 1
            oneHotRed = 0
        else:
            oneHotWhite = 0
            oneHotRed = 1

        # print(str(words[11].rstrip()) + " White: " + str(oneHotWhite) + " Red: " + str(oneHotRed))

        # fixed = [0], volatile = [1], citric = [2],
        # residual = [3], chlorides = [4], free = [5], total = [6],
        # density = [9], ph = [8], sulphates = [9], alcohol = [10], wineType = [11]
        wine = Wine(words[0], words[1], words[2], words[3], words[4], words[5], words[6],
                    words[7], words[8], words[9], words[10], oneHotWhite, oneHotRed)
        wineryToCreate.addToCatalog(wine)

    return wineryToCreate

# ....................... END OF WINERY CREATOR MAIN METHOD.......................

# matrixCreator gets a winery catalog and convert its wines to a matrix.
def matrixCreator(wineryToConvert):
    matrix = []
    for wineToConvert in wineryToConvert.catalog:
        lineInMatrix = [wineToConvert.fixed, wineToConvert.volatile, wineToConvert.citric, wineToConvert.residual,
                        wineToConvert.chlorides, wineToConvert.free, wineToConvert.total, wineToConvert.density,
                        wineToConvert.ph, wineToConvert.sulphates, wineToConvert.alcohol,
                        wineToConvert.white, wineToConvert.red]
        matrix.append(lineInMatrix)

    return matrix

# ..................................... END OF matrixCreator MAIN FUNCTION........................................

# matrixWithY function get couples of values and a label and merge them.
def matrixWithY(values, labels):
    result = []
    for index in range(len(values)):
        production = [values[index], labels[index]]
        result.append(production)
    return result
# ..................................... END OF matrixWithY MAIN FUNCTION...........................................

# WeightsGeneratorFunction gets the number of features and number of possible labels, and create a zeroed function.
def weightGeneratorFunction(numberOfLabels, featuresNumber):
    weights = []
    for _ in range(numberOfLabels):
        weight = []
        for _ in range(featuresNumber):
            weight.append(0)
        weights.append(weight)
    return weights
# ......................END OF WeightsGeneratorFunction MAIN FUNCTION.....................................

# innerProductCalc gets a product and calculates its value.
def innerProductCalc(example, weight):
    outputResult = 0
    for index in range(len(weight)):
        production = example[index] * weight[index]
        outputResult += production
    return outputResult

#......................END OF innerProductionCalc MAIN FUNCTION.....................................

# predictLabel gets example and a weight function and predicts its label.
def predictLabel(example, weight):
    innerProducts = []
    for myIndex in range(len(weight)):
        innerProduct = innerProductCalc(example, weight[myIndex])
        innerProducts.append(innerProduct)
    # print(innerProducts)
    argMax = innerProducts.index(max(innerProducts))
    return [argMax, innerProducts]  # Returning both the argmax and products for PA.

#......................END OF predictLabel MAIN FUNCTION.....................................

# calcEuc is a function that calculates the Euclidean distance for two wines.
def calcEuc(wine, trainingWine):
    eucDistance = math.sqrt(((float(wine.fixed) - float(trainingWine.fixed)) ** 2) +
                            ((float(wine.volatile) - float(trainingWine.volatile)) ** 2) +
                            ((float(wine.citric) - float(trainingWine.citric)) ** 2) +
                            ((float(wine.residual) - float(trainingWine.residual)) ** 2) +
                            ((float(wine.chlorides) - float(trainingWine.chlorides)) ** 2) +
                            ((float(wine.free) - float(trainingWine.free)) ** 2) +
                            ((float(wine.total) - float(trainingWine.total)) ** 2) +
                            ((float(wine.density) - float(trainingWine.density)) ** 2) +
                            ((float(wine.ph) - float(trainingWine.ph)) ** 2) +
                            ((float(wine.sulphates) - float(trainingWine.sulphates)) ** 2) +
                            ((float(wine.alcohol) - float(trainingWine.alcohol)) ** 2) +
                            ((float(wine.white) - float(trainingWine.white)) ** 2) +
                            ((float(wine.red) - float(trainingWine.red)) ** 2))

    return eucDistance

# ..................................... END OF calcEuc MAIN FUNCTION..................................................

# knn function predicts the y values of the example set based on the training set using K Nearest Neighbors (KNN)
def knn(trainingWinery, trainingY, exampleWinery, k):
    exampleY = []

    for wine in exampleWinery.catalog:
        eucArray = np.array([])
        for trainingWine in trainingWinery.catalog:
            eucDistance = calcEuc(wine,
                                  trainingWine)  # Using calcEuc, we're calculating the Euclidean distance for the inner and outer wines.
            eucArray = np.append(eucArray, eucDistance)

        myNeighbors = np.argpartition(eucArray,
                                      k)  # Find the indices of the k min distances in a fashion similar to algorithm Select.
        myNeighbors = myNeighbors[:k]  # Filter all unnecessary indices
        neighborsY = (np.array([])).astype(
            int)  # By default npArray is of type float, we need it to contain ints for bincount.
        for index in range(k):
            neighborsY = np.append(neighborsY, (trainingY[myNeighbors[index]]))
        # print(neighborsY)
        if(k > 0):
            majority = np.bincount(neighborsY).argmax()  # Get the most frequent Y value of the neighbors.
            exampleY.append(majority)

    return exampleY

# ....................................... END OF knn MAIN FUNCTION...........................................

# perceptron function predicts the y values of the example set based on the training set using Multiclass Perceptron.
def perceptron(trainingWinery, trainingY, featuresNumber, epochs, numberOfLabels, eta):
    weights = weightGeneratorFunction(numberOfLabels, featuresNumber)

    result = matrixWithY(trainingWinery, trainingY)
    minMistakes = 20000000
    bestEpoch = 0
    currentEpoch = 0
    averageMistakes = 0

    for _ in range(epochs):
        mistakes = 0

        for example, label in result:
            y_hat = predictLabel(example, weights)  # Getting argmax AND inner products.
            y_hat = y_hat[0]  # Extracting the argmax value.
            # Resource: https://www.youtube.com/watch?v=aYAVvXVVXOw
            # print("y_hat: " + str(y_hat) + " label: " + str(label))
            if(y_hat != label):
                mistakes += 1
                for index in range(len(weights[0])):  # Updating the correct label and the actual predicted label weight functions.
                    weights[label][index] += eta * example[index]  # Using a considerate small eta.
                    weights[y_hat][index] -= eta * example[index]

        if(mistakes < minMistakes):
            minMistakes = mistakes
            bestEpoch = currentEpoch

        averageMistakes += mistakes
        currentEpoch += 1

    averageMistakes /= epochs
    # print("Lowest Number Of Mistakes: " + str(minMistakes) + " Happened in Epoch: " + str(bestEpoch) + " Average number Of Mistakes:" + str(averageMistakes))
    return weights

# ....................................... END OF perceptron MAIN FUNCTION...........................................

# lossCalculator calculates the loss function for PA model.
def lossCalculator(weights, example, label, y_hat):
    # print(innerProductCalc(example, weights[y_hat]))
    # print(max([0, 1 - innerProductCalc(example, weights[label]) + innerProductCalc(example, weights[y_hat])]))
    return max([0, 1 - innerProductCalc(example, weights[label]) +
                innerProductCalc(example, weights[y_hat])])

# ........................... END OF lossCalculator MAIN FUNCTION .................................

# tauCalculator, calculates the tau value for the Passive Aggressive model
def tauCalculator(example, loss):
    # Reminder: Tau = (loss of x transpose, label transpose, weight transpose) / 2 * norm x transpose ^ 2
    product = innerProductCalc(example, example)
    # print(product)
    if(product == 0):
        return 0
    return (loss / (2 * product))

# ........................... END OF tauCalculator MAIN FUNCTION .................................

# PassiveAgressiveUpdate function, calculates the update to our model. if either loss or tau = 0, no update will be made.
def passiveAggressiveUpdate(weights, example, label, y_hat):
    loss = lossCalculator(weights, example, label, y_hat)
    tau = tauCalculator(example, loss)
    for index in range(len(weights[0])):
        weights[label][index] += tau * example[index]
        weights[y_hat][index] -= tau * example[index]

# ........................... END OF passiveAggressiveUpdate MAIN FUNCTION .................................

# passiveAggressiveWeight calculates weight function based on the Passive Aggressive model.
def passiveAggressiveWeight(trainingWinery, trainingY, featuresNumber, epochs, numberOfLabels):
    weights = weightGeneratorFunction(numberOfLabels, featuresNumber)
    result = matrixWithY(trainingWinery, trainingY)

    for _ in range(epochs):

        for example, label in result:
            y_hat = predictLabel(example, weights)
            # y_hat[0] a number, y_hat[1] a list.
            passiveAggressiveUpdate(weights, example, label, y_hat[0])

    return weights

# ................................ END OF passiveAggressiveWeight MAIN FUNCTION....................................


numberOfFeatures = 13
cuWinery = Winery()  # To save x values of Training

cuMinMaxWinery = Winery()
cuMeanWinery = Winery()
cuZScoreWinery = Winery()

cuExampleWinery = Winery()  # To save x values of Examples

cuExampleMinMaxWinery = Winery()
cuExampleZScoreWinery = Winery()

yScoresOfTraining = []  # To save y values of Training

knnPredictionArray = []

# Building the database.
# file = open(sys.argv[1], "r+")
# train_x.txt train_y.txt test_x.txt
file = open("train_x.txt", "r+")
file.seek(0)
linesOfFile = file.readlines()
cuWinery = wineryCreator(linesOfFile)
file.close()

# file2 = open(sys.argv[2], "r+")
file2 = open("train_y.txt", "r+")
file2.seek(0)
linesOfFile = file2.readlines()
for line in linesOfFile:
    line = int(line)
    yScoresOfTraining.append(line)

file2.close()

setOfLabels = list(set(yScoresOfTraining))  # Preserving the number of labels we have.

# file3 = open(sys.argv[3], "r+")
file3 = open("test_x.txt", "r+")
file3.seek(0)
linesOfFile = file3.readlines()
cuExampleWinery = wineryCreator(linesOfFile)
file3.close()

cuWineTester = WineTester(cuWinery)  # To get statistics on my training set.
cuExampleTester = WineTester(cuExampleWinery)  # To get statistics on my example set.

cuWineTester.minMaxNormalization(cuMinMaxWinery)
cuWineTester.meanNormalization(cuMeanWinery)
cuWineTester.zScoreNormalization(cuZScoreWinery)

cuExampleTester.zScoreNormalization(cuExampleZScoreWinery)
cuExampleTester.minMaxNormalization(cuExampleMinMaxWinery)
# cuTestWinery.zScoreNormalization(cuTestZScoreWinery)

kFoldArrayY = []
kFoldWinery = Winery()
for index in range(50):
    kFoldWinery.catalog.append(cuWinery.catalog[index])
    kFoldArrayY.append(yScoresOfTraining[index])

# print(str(len(kFoldArrayX)) + " " + str(len(kFoldArrayY)))

kFoldWineTester = WineTester(kFoldWinery)
kFoldZScoreWinery = Winery()
kFoldWineTester.minMaxNormalization(kFoldZScoreWinery)

k = 2
knnPredictionArray = knn(cuMinMaxWinery, yScoresOfTraining, cuExampleMinMaxWinery, k)  # With normalization

cuZScoreWineryMatrix = matrixCreator(cuZScoreWinery)
cuMinMaxWineryMatrix = matrixCreator(cuMinMaxWinery)

cuExampleMinMaxWineryMatrix = matrixCreator(cuExampleMinMaxWinery)

epochs = 958
eta = 0.01
perceptronWeights = perceptron(cuMinMaxWineryMatrix, yScoresOfTraining, numberOfFeatures, epochs, len(setOfLabels), eta)

perceptronPredictionArray = []
for rowInMatrix in cuExampleMinMaxWineryMatrix:
    perceptronPredictionArray.append(predictLabel(rowInMatrix, perceptronWeights)[0])

epochs = 780
paWeights = passiveAggressiveWeight(cuMinMaxWineryMatrix, yScoresOfTraining, numberOfFeatures, epochs, len(setOfLabels))
paPredictionArray = []
for rowInMatrix in cuExampleMinMaxWineryMatrix:
    paPredictionArray.append(predictLabel(rowInMatrix, paWeights)[0])

for index in range(len(paPredictionArray)):  # Printing using the desired pattern.
    print("knn:" + str(knnPredictionArray[index]) + ", perceptron: " + str(perceptronPredictionArray[index]) +
          ", pa: " + str(paPredictionArray[index]))

# .................................... QUALITY CHECKS ...........................................................

# Searching for the best k KNN.
'''bestK = 0
highestPrecision = 0
for k in range(100):
    hits = 0
    attempts = 0

    if k == 0:
        continue

    knnPredictionArray = knn(cuZScoreWinery, yScoresOfTraining, kFoldZScoreWinery, k)
    for index in range(len(kFoldArrayY)):
        if knnPredictionArray[index] == kFoldArrayY[index]:
            hits += 1
        attempts += 1
    precision = hits / attempts
    print(precision)
    if(precision > highestPrecision):
        bestK = k
        highestPrecision = precision

print("Best k for KNN is: " + str(bestK) + " with a precision of: " + str(highestPrecision))'''

'''# Searching for best epoch and eta for Perceptron
bestEta = 0
bestEpoch = 0
highestPrecision = 0
kFoldMatrix = matrixCreator(kFoldZScoreWinery)
for epoch in range(1000):
    hits = 0
    attempts = 0
    if epoch == 0:
        continue
    eta = 0.01
    perceptronWeights = perceptron(cuZScoreWineryMatrix, yScoresOfTraining, numberOfFeatures, epoch,
                                       len(setOfLabels), eta)
    perceptronPredictionArray = []
    for rowInMatrix in kFoldMatrix:
        perceptronPredictionArray.append(predictLabel(rowInMatrix, perceptronWeights)[0])
        # print(perceptronPredictionArray)
    for index in range(len(kFoldArrayY)):
        if perceptronPredictionArray[index] == kFoldArrayY[index]:
            hits += 1
        attempts += 1
    precision = hits / attempts
    print(precision)
    if precision > highestPrecision:
        bestEta = eta
        bestEpoch = epoch
        highestPrecision = precision

print("Best eta for Perceptron is: " + str(bestEta) + " Best number of epochs are: " + str(bestEpoch) 
+ " With a precision of: " + str(highestPrecision))'''

# Searching for best epoch for PA
'''bestEpoch = 0
highestPrecision = 0
kFoldMatrix = matrixCreator(kFoldZScoreWinery)
for epoch in range(1000):
    if epoch == 0:
        continue
    hits = 0
    attempts = 0
    paWeights = passiveAggressiveWeight(cuMinMaxWineryMatrix, yScoresOfTraining, numberOfFeatures, epoch,
                                        len(setOfLabels))
    # print(paWeights)
    paPredictionArray = []
    for rowInMatrix in kFoldMatrix:
        paPredictionArray.append(predictLabel(rowInMatrix, paWeights)[0])
    for index in range(len(kFoldArrayY)):
        if paPredictionArray[index] == kFoldArrayY[index]:
            hits += 1
        attempts += 1
    precision = hits / attempts
    print(precision)
    if precision > highestPrecision:
        highestPrecision = precision
        bestEpoch = epoch

print("Best Epoch is: " + str(bestEpoch) + "With a precision of: " + str(highestPrecision))'''
