from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsummary import summary

from resnet import *
from utils import Basics

# Defining constants, based on assignment
OUTPUT_CLASSES = 10
MEANS = (0.4914, 0.4822, 0.4465)
STD_DEV = (sqrt(0.2023), sqrt(0.1994), sqrt(0.2010))
IO_PROCESSES = 2
FLIP_PROBABILITY = 0.5
TRAINING_BATCH_SIZE = 64
TESTING_BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
ROOT = "./.data"
MAX_PARAMS = 5000000

# Lets start by defning transforms.
# I'm going to assume that transforms are a part of the learning process
# This implies that no transformations will be applied to the testing data
# Except the normalization step obviously
training_transformations = transforms.Compose(
    [
        transforms.RandomCrop(
            size=(32, 32), padding=4),
        transforms.RandomHorizontalFlip(
            FLIP_PROBABILITY),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STD_DEV)
    ]
)

testing_transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STD_DEV)
    ]
)


def downloadData(ROOT):
    # Now we load training amd testing data
    # Note, this is different to the DataLoader step. That shall come after this
    training_data = datasets.CIFAR10(
        ROOT,
        train=True,
        download=True,
        transform=training_transformations
    )

    testing_data = datasets.CIFAR10(
        ROOT,
        train=False,
        download=True,
        transform=testing_transformations
    )

    return training_data, testing_data


if __name__ == "__main__":

    trainingData, testingData = downloadData(ROOT)

    trainingDataLoader = torch.utils.data.DataLoader(
        dataset=trainingData,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True
    )
    testingDataLoader = torch.utils.data.DataLoader(
        dataset=testingData,
        batch_size=TESTING_BATCH_SIZE,
        shuffle=True
    )
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")

    model = ResNet18_mod().to(device)

    modelParams = model.parameters()
    setParams = {
        "params": modelParams,
        "lr": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "nesterov": True,
    }

    optimizer = optim.SGD(**setParams)
    lossFunction = nn.CrossEntropyLoss()

    trainingAbstraction = Basics(
        model, optimizer, lossFunction, trainingDataLoader, testingDataLoader)

    parameters = trainingAbstraction._countParameters()
    print(f"Model has {parameters} parameters")
    assert parameters <= MAX_PARAMS
    print(f"You have {MAX_PARAMS-parameters} left!")
    summary(model, (3, 32, 32))

    trainingAbstraction.trainEpochs()
