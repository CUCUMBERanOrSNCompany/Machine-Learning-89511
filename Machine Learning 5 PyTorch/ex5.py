# Machine Learning Assignment 5
# DEVELOPED BY Tomer Himi & CUCUMBER AN OrSN COMPANY.
# UNAUTHORIZED COPY OF THIS WORK IS STRICTLY PROHIBITED.
# DEVELOPED FOR EDUCATIONAL PURPOSES, FOR THE COURSE MACHINE LEARNING 89511.
# BAR ILAN UNIVERSITY, JANUARY, 2021.
# ALL RIGHTS RESERVED.

# Imports From gcommand_dataset.py
import os
import os.path

import soundfile as sf
import librosa
import torch.utils.data as data

# ....................................

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.nn.init import xavier_uniform_ as xavier

# Gcommand_dataset.py code

AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    spects = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_audio_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    spects.append(item)
    return spects


def spect_loader(path, window_size, window_stride, window, normalize, max_len=101):
    y, sr = sf.read(path)
    # n_fft = 4096
    n_fft = int(sr * window_size)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    # S = log(S+1)
    spect = np.log1p(spect)

    # make all spects with the same dims
    # TODO: change that in the future
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect


class GCommandLoader(data.Dataset):
    """A google command data set loader where the wavs are arranged in this way: ::
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, transform=None, target_transform=None, window_size=.02, # Constructor of class GCommand
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        # print("Entered init")
        classes, class_to_idx = find_classes(root)
        spects = make_dataset(root, class_to_idx)

        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

    # Fields

        self.root = root
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len
        self.len = len(self.spects)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        # print(index)
        path, target = self.spects[index]
        spect = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)
        # print (path)
        if self.transform is not None:
            spect = self.transform(spect)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return spect, target

    def __len__(self):
        return self.len


# Our code from the previous assignment
'''
class AToBNetworks(nn.Module):
    # class for the first two models A and B, using RelU in forward
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
        return F.log_softmax(x, dim=1)
'''


'''
class CNetwork(nn.Module):
    # class for the third model using RelU and dropout regularization
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
        x = self.do0(x)  # Regularization AFTER - do0 stands for Dropout0
        x = F.relu(self.fc1(x))
        x = self.do1(x)  # Regularization AFTER - do1 stands for Dropout1
        return F.log_softmax(self.fc2(x), dim=1)


class DNetwork(nn.Module):
    # class for the forth model using RelU and batch normalization
    def __init__(self, image_size, fc0_size=100, fc1_size=50, fc2_size=10):
        super(DNetwork, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, fc0_size)
        self.bn0 = nn.BatchNorm1d(num_features=fc0_size)  # nn.BatchNorm1d performs Batch normalization.
        self.fc1 = nn.Linear(fc0_size, fc1_size)
        self.bn1 = nn.BatchNorm1d(num_features=fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn2 = nn.BatchNorm1d(num_features=fc2_size)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        return F.log_softmax(self.bn2(self.fc2(x)), dim=1)
'''
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.firstConvLayer = nn.Conv2d(1, 5, 5)  # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html - New size: 157 X 97: (Width or height - filter size)/stride + 1
        self.batchNorm1 = nn.BatchNorm2d(5)
        self.poolingOfFirst = nn.MaxPool2d(2, 2)  # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html, 1st arg: Filter, 2nd Arg: Stride
        self.secondConvLayer = nn.Conv2d(5, 10, 5)
        self.batchNorm2 = nn.BatchNorm2d(10)
        self.poolingOfSecond = nn.MaxPool2d(2, 2)  # Consider to omit!
        self.fullyConnected1 = nn.Linear(37 * 22 * 10, 100)  # After CNN, size is 37 * 22 * 10
        self.batchNorm3 = nn.BatchNorm1d(100)
        self.fullyConnected2 = nn.Linear(100, 50)
        self.batchNorm4 = nn.BatchNorm1d(50)
        self.fullyConnected3 = nn.Linear(50, 30)
        self.batchNorm5 = nn.BatchNorm1d(30)

    def forward(self, example):
        # CNN with Pooling stage
        example = self.firstConvLayer(example)
        example = self.batchNorm1(example)
        example = F.relu(example)
        example = self.poolingOfFirst(example)
        example = self.secondConvLayer(example)
        example = self.batchNorm2(example)
        example = F.relu(example)
        example = self.poolingOfFirst(example)

        # Fully Connected with Batch Normalization stage
        example = example.view(-1, 37 * 22 * 10)  # todo: We're still trying to figure out that additional parameter.
        example = self.fullyConnected1(example)
        example = self.batchNorm3(example)
        example = F.relu(example)
        example = self.fullyConnected2(example)
        example = self.batchNorm4(example)
        example = F.relu(example)
        example = self.fullyConnected3(example)
        example = self.batchNorm5(example)

        # print("Hi")

        return F.log_softmax(example, dim=1)

'''
class EToFNetworks(nn.Module):
    # class for the final two models E and F using either RelU or Sigmoid and ADAM or SGD
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
        if self.activation == "relu":  # For Fifth model
            x = F.relu(self.fc0(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))

        elif self.activation == "sigmoid":  # RELEVANT TO MODEL 6 ONLY
            x = torch.sigmoid(self.fc0(x))
            x = torch.sigmoid(self.fc1(x))
            x = torch.sigmoid(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            x = torch.sigmoid(self.fc5(x))
        return F.log_softmax(x, dim=1)

'''

def train(model):
    """Train function, trains our model using the ordinary drill of Forward, Backwards etc.
    After each epoch of training, validation step has activated"""
    model.train()
    correct = 0
    train_loss = 0

    for batch_idx, (data, labels) in enumerate(train_loader):  # train step
        optimizer.zero_grad()
        output = model(data)  # performs Forward apparently
        loss = F.nll_loss(output, labels)
        train_loss += F.nll_loss(output, labels, reduction='sum').item()
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)

def validation(model):
    val_loss = 0
    correct = 0

    for data, labels in validation_loader:  # val step
        output = model(data)
        val_loss += F.nll_loss(output, labels, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum().item()

    val_loss /= len(validation_loader.dataset)
    val_accuracy = 100. * correct / len(validation_loader.dataset)

def test(model):
    """Test function, helps us predict the label of each example of fashion MNIST (test from PyTorch)"""
    model.eval()
    test_loss = 0
    correct = 0
    index = 0
    file = open("test_y", "w+")

    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data)  # Performs Forward apparently
            test_loss += F.nll_loss(output, labels, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            print(str(index) + ".wav," + newDictionary.get(pred))
            file.write(str(index) + ".wav," + newDictionary.get(pred))
            index += 1
            correct += pred.eq(labels.view_as(pred)).cpu().sum().item()
        test_loss /= len(test_loader.dataset)
        print(str(correct / index))
    file.close()

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)

if __name__ == "__main__":
    datasetOfTrain = GCommandLoader('ex5_data/train')  # Converting audio files to images.
    train_loader = torch.utils.data.DataLoader(datasetOfTrain, batch_size=64, shuffle=True, pin_memory=True)

    datasetOfValid = GCommandLoader('ex5_data/valid')
    validation_loader = torch.utils.data.DataLoader(datasetOfValid, batch_size=64, shuffle=True, pin_memory=True)

    datasetOfTest = GCommandLoader('ex5_data/test')
    test_loader = torch.utils.data.DataLoader(datasetOfTest, batch_size=64, shuffle=False, pin_memory=True)

    newDictionary = dict([(value, key) for key, value in datasetOfTrain.class_to_idx.items()])

    # the best model to run
    # model = DNetwork(image_size=28 * 28)
    model = Network()
    model.apply(weight_init)  # apply is a saved term in Python which refer to another function that is passed in the argument.
    # In other words, there ISN'T a function named "apply", rather a function named weight_init.
    # https://python-reference.readthedocs.io/en/latest/docs/functions/apply.html

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # lr stands for Learning Rate. torch.optim injects an OPTIMIZER.
    epoch = 10


    # train and validation steps
    for _ in range(epoch):
        train(model)  # todo: Rewrite this function!
        validation(model)  # todo: Write this function!

    test(model)  # Outside of FOR LOOP deliberately. todo: review this function!
