# Machine Learning Assignment 1
# DEVELOPED BY CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, NOVEMBER, 2020.
# ALL RIGHTS RESERVED.

import sys
import math
import numpy as np
import scipy.io.wavfile


# Class to define a single point from the sample.

class Point:
    def __init__(self, xCordinate, yCordinate):
        self.xCordinate = xCordinate
        self.yCordinate = yCordinate
        self.belongsToCreed = -1

    def getX(self):
        return self.xCordinate

    def getY(self):
        return self.yCordinate

    def setX(self, newX):
        self.xCordinate = newX

    def setY(self, newY):
        self.yCordinate = newY


# .........................................

# Class to define a single centroid.
class Centroid:

    def __init__(self, xCordinate, yCordinate):
        self.xCordinate = xCordinate
        self.yCordinate = yCordinate
        self.creed = []  # To save the points associated to the centroid.

    def getX(self):
        return self.xCordinate

    def getY(self):
        return self.yCordinate

    def getCreed(self):
        return self.creed

    def setX(self, newX):
        self.xCordinate = newX

    def setY(self, newY):
        self.yCordinate = newY

    def addToCreed(self, point):
        self.creed.append(point)

    def expelFromCreed(self, point):
        self.creed.remove(point)


# ...........................................

# Class that holds all points from the sample
class Points:
    def __init__(self):
        self.points = []

    def addToPoints(self, point):
        self.points.append(point)

    def addToCentroidCreed(self, listOfCentroids):
        for point in self.points:
            index = 0
            minimumDistance = math.sqrt(((point.xCordinate - listOfCentroids.centroids[0].xCordinate) ** 2) +
                                        ((point.yCordinate - listOfCentroids.centroids[0].yCordinate) ** 2))
            key = 0

            for centroid in listOfCentroids.centroids:

                distance = math.sqrt(((point.xCordinate - listOfCentroids.centroids[index].xCordinate) ** 2) +
                                     ((point.yCordinate - listOfCentroids.centroids[index].yCordinate) ** 2))

                if (distance < minimumDistance):
                    key = index
                    minimumDistance = distance

                index = index + 1

            # print(key)
            if (point.belongsToCreed != -1):
                listOfCentroids.centroids[point.belongsToCreed].expelFromCreed(point)

            listOfCentroids.centroids[key].addToCreed(point)
            point.belongsToCreed = key
            # print("Winning centroid:\n" + str(listOfCentroids.centroids[key]) + "\n" +
            #     str(listOfCentroids.centroids[key].creed) + "...\n") #todo: REMOVE BEFORE PUBLISHING!


# ...........................................


# Class that holds all centroids
class Centroids:
    def __init__(self):
        self.centroids = []

    def addToCentroids(self, centroid):
        self.centroids.append(centroid)

    def showCentroids(self):
        centroidCoordinates = ""
        counter = 0
        for centroidInList in self.centroids:

            x = centroidInList.getX()
            y = centroidInList.getY()
            x = str(x)
            x = x[:-1]
            y = str(y)
            y = y[:-1]

            if (counter != 0):
                centroidCoordinates = centroidCoordinates + ","

            counter = counter + 1

            centroidCoordinates = centroidCoordinates + "[" + x + " " + y + "]"

        return centroidCoordinates

    def calculateNewCordinates(self):
        somethingChanged = 0
        for centroidInList in self.centroids:
            x = 0
            y = 0
            for pointInCreed in centroidInList.creed:
                x = x + pointInCreed.getX()
                y = y + pointInCreed.getY()

            if (len(centroidInList.creed) > 0):
                x = x / len(centroidInList.creed)
                x = float(x.round())
                y = y / len(centroidInList.creed)
                y = float(y.round())

                if ((x != centroidInList.getX()) | (y != centroidInList.getY())):
                    centroidInList.setX(x)
                    centroidInList.setY(y)
                    somethingChanged = 1

        return somethingChanged

    def calculateCost(self, numberOfPoints):
        totalCost = 0
        for centroidInList in self.centroids:
            for pointInCreed in centroidInList.creed:
                totalCost = totalCost + (((pointInCreed.xCordinate - centroidInList.xCordinate) ** 2) +
                                         ((pointInCreed.yCordinate - centroidInList.yCordinate) ** 2))

        # print(totalCost)
        totalCost = totalCost / numberOfPoints
        return (float(totalCost.round()))

    # A method that changes the value of each point in the audio file to its centroid's values.
    def pointsCompressor(self):
        for centroidInList in self.centroids:
            for pointInCreed in centroidInList.creed:
                pointInCreed.setX(centroidInList.xCordinate)
                pointInCreed.setY(centroidInList.yCordinate)


# ..................MAIN..........................
somethingChanged = 1

file = open("output.txt", "w+")
file.seek(0)

sample, centroids = sys.argv[1], sys.argv[2]
fs, y = scipy.io.wavfile.read(sample)

x = np.array(y.copy())
centroids = np.loadtxt(centroids)

cuCentroids = Centroids()
cuPoints = Points()
numberOfChanges = 0

for rawCentroid in centroids:
    centroid = Centroid(rawCentroid[0], rawCentroid[1])
    cuCentroids.addToCentroids(centroid)

for rawPoint in x:
    point = Point(rawPoint[0], rawPoint[1])  # rawPoint[0] = X, rawPoint[1] = Y
    cuPoints.addToPoints(point)

print(len(cuPoints.points))


while ((somethingChanged > 0) & (numberOfChanges <= 30)):
    returnValue = cuCentroids.showCentroids()
    # print(returnValue)
    cuPoints.addToCentroidCreed(cuCentroids)
    cuCentroids.calculateCost(len(cuPoints.points))
    somethingChanged = cuCentroids.calculateNewCordinates()
    #print("..............." + str(numberOfChanges) + ".....................")
    if (numberOfChanges > 0):
        file.write("[iter " + str(numberOfChanges - 1) + "]:" + returnValue + "\n")
    numberOfChanges = numberOfChanges + 1

if ((numberOfChanges <= 30) & (somethingChanged == 0)):
    file.write("[iter " + str(numberOfChanges - 1) + "]:" + returnValue + "\n")
    numberOfChanges = numberOfChanges + 1

#file.write("For: " + str(len(cuCentroids.centroids)) + "Centroids\n")
#while (somethingChanged > 0):
 #   cuPoints.addToCentroidCreed(cuCentroids)
  #  cost = cuCentroids.calculateCost(len(cuPoints.points))
   # file.write("Iteration: " + str(numberOfChanges + 1) + ": " + str(cost) + "\n")
    #somethingChanged = cuCentroids.calculateNewCordinates()
    #numberOfChanges = numberOfChanges + 1

cuCentroids.pointsCompressor()

index = 0

newValues = x

for rawPoint in newValues:
    rawPoint[0] = int(cuPoints.points[index].getX())
    rawPoint[1] = int(cuPoints.points[index].getY())
    index = index + 1

x = np.asarray(x, dtype=np.int16)

scipy.io.wavfile.write("compressed.wav", fs, np.array(newValues, np.int16))

file.close()
