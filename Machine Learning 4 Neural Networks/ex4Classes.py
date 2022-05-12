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
from torchvision import transforms
from torchvision import datasets

# Converting a database to Tensors.
class Convertor(nn.Module):
    def __init__(self, database):
        self.database = database
        self.get()

    def getLength(self, database):
        return len(database)

    def get(self):
        self.database = torch.tensor(self.database)


class AToBNetworks(nn.Module):
    def __init__(self, image_size):
        super(AToBNetworks, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)  # Input -> 1st
        self.fc1 = nn.Linear(100, 50)  # 1st -> 2nd
        self.fc2 = nn.Linear(50, 10)  # 2nd -> Output

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        result = F.log_softmax(x)
        return result

class CNetwork(nn.Module):
    def __init__(self, image_size, fc0_size=100, fc1_size=50, fc2_size=10):
        super(CNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.do0 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.do0(x)  # Regularization AFTER
        x = F.relu(self.fc1(x))
        x = self.do1(x)  # Regularization AFTER
        return F.log_softmax(self.fc2(x))

class DNetwork(nn.Module):
    def __init__(self, image_size, fc0_size=100, fc1_size=50, fc2_size=10):
        super(DNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = nn.BatchNorm1d(num_features=x)
        x = F.relu(x)
        x = self.fc1(x)
        x = nn.BatchNorm1d(num_features=x)
        x = F.relu(x)
        x = self.fc2(x)
        x = nn.BatchNorm1d(num_features=x)
        return F.log_softmax(x)

class EToFNetworks(nn.Module):
    def __init__(self, image_size, activation):
        super(EToFNetworks, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)  # Input -> 1st
        self.fc1 = nn.Linear(128, 64)  # 1st -> 2nd
        self.fc2 = nn.Linear(64, 10)  # 2nd -> 3rd
        self.fc3 = nn.Linear(10, 10)  # 3rd -> 4th
        self.fc4 = nn.Linear(10, 10)  # 4th -> 5th
        self.fc5 = nn.Linear(10, 10)  # 5th -> Output
        self.activation = activation

    def forward(self, x):
        x = x.view(-1, self.image_size)
        if self.activation == "relu":
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))

        elif self.activation == "sigmoid":  # RELEVANT TO MODEL 6 ONLY
            x = F.sigmoid(self.fc0(x))
            x = F.sigmoid(self.fc1(x))
            x = F.sigmoid(self.fc2(x))
            x = F.sigmoid(self.fc3(x))
            x = F.sigmoid(self.fc4(x))
            x = F.sigmoid(self.fc5(x))
        return F.log_softmax(x)

# ............................................ END OF CLASSES ........................................................

def train(epoch, model):
    model.train()  # Model is one of the four classes above (AToB, C, D or EToF)
    # print(type(model))
    for _ in range(epoch):
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)  # Performs Forward apparently
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
        print(float(loss.data))

def trainValidation(epoch, model):
    model.train()
    results = []
    counter = 0  # To help us determine how many epochs passed, and if we need to begin recording the results.
    # We're doing so, only at the LAST EPOCH
    for _ in range(epoch):
        counter += 1
        for batch_idx, (data, labels) in enumerate(test_loader):
            output = model(data)  # Performs Forward apparently
            if counter == epoch:
                for index in range(len(output)):
                    results.append(torch.exp(output[index]))
            # output = torch.exp(output)
            # output = torch.argmax(output)
            # print(output)
            loss = F.nll_loss(output, labels)
            # loss = F.nll_loss(labels, output)
        print(float(loss.data))
    return results

def argMaxCalc(arrayOfProbabilities):
    results = []
    for line in arrayOfProbabilities:
        results.append(float(torch.argmax(line)))
    return results

def accuracyCalculator(arrayOfPredictions, arrayOfLabels):
    hits = 0
    index = 0

    while index < len(arrayOfPredictions):
        if arrayOfPredictions[index] == arrayOfLabels[index]:
            hits += 1
        index += 1

    result = hits / len(arrayOfLabels)
    return result

# .................................................. MAIN .............................................................

debugLimit = 500  # To limit the number of examples during the building
# trainX = np.loadtxt(sys.argv[1], max_rows=debugLimit) / 255
# trainY = np.loadtxt(sys.argv[2], max_rows=debugLimit)

transforms = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])

fashion = datasets.FashionMNIST("./data", train=True, download=True, transform=transforms)
train_set, val_set = torch.utils.data.random_split(fashion, [round(len(fashion)*0.8), len(fashion)-round(len(fashion)*0.8)])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True)


# print(train_set.dataset.train_labels)


testX = np.loadtxt(sys.argv[3], max_rows=debugLimit) / 255
tensorTestX = Convertor(testX)

# classTester = Convertor(trainX)

# validationSetSize = len(trainX) * 0.2
# validationSetSize = math.floor(validationSetSize)

# validationTrainX = trainX[0: validationSetSize]
# trainTrainX = trainX[validationSetSize: len(trainX)]
# tensorValidationTrainX = Convertor(validationTrainX)  # Creating Tensors from Numpy array
# tensorTrainTrainX = Convertor(trainTrainX)  # Creating Tensors from Numpy array

# validationTrainY = trainY[0: validationSetSize]
# trainTrainY = trainY[validationSetSize: len(trainY)]
# tensorValidationY = Convertor(validationTrainY)  # Creating Tensors from Numpy array
# tensorTrainTrainY = Convertor(trainTrainY)  # Creating Tensors from Numpy array

# xYData = zip(tensorTrainTrainX.database, tensorTrainTrainY.database)

model = AToBNetworks(image_size=28*28)

optimizerOpt = "SGD"
learningRate = 0.01
epoch = 10

if optimizerOpt == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
elif optimizerOpt == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# yHat = train(epoch, model)
yHatProbabilities = trainValidation(epoch, model)
# print(len(yHatProbabilities))
yHat = argMaxCalc(yHatProbabilities)

# print(torch.argmax(yHatProbabilities[0]))
# print(len(yHat))
# todo: Apply argmax on yHat
accuracy = accuracyCalculator(yHat, val_set.dataset.train_labels)
# accuracy = accuracyCalculator(yHat, train_set.dataset.train_labels)
print(accuracy)
print("Hi")


# validationTrainXInTensors = torch.from_numpy(validationTrainX)
# validationTrainYInTensors = torch.from_numpy(validationTrainY)

# trainTrainXInTensors = torch.from_numpy(trainTrainX)
# trainTrainYInTensors = torch.from_numpy(trainTrainY)

# testXInTensors = torch.from_numpy(testX)
