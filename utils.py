"""
Authors:
Mihir Upasani (mu2047@nyu.edu)
Shivani Gawande
Fabiha Khalid
"""
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


class Basics():

    def __init__(self, model, optimizer, schedule, criterion, training_dl, testing_dl, modelName):
        self.model = model
        self.optimizer = optimizer
        self.schedule = schedule
        self.criterion = criterion
        self.device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.trainingHistory = {
            'accuracy': [],
            'loss': [],
        }
        self.testingHistory = {
            'accuracy': [],
            'loss': [],
        }
        self.validationHistory = {
            'accuracy': [],
            'loss': [],
        }
        self.trainingDataLoader = training_dl
        self.testingDataLoader = testing_dl
        self.modelName = modelName

    def __calculateAccuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc

    def __getTime(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _countParameters(self):
        total_params = 0
        for name, parameter in self.model.named_parameters():
            params = parameter.numel()
            total_params += params
        return total_params

    def _trainModel(self, epoch):
        """
        Abstraction layer for training steps
        1. Make predictions
        2. Calculate loss, accuracy
        3. Propogate loss backwards and update weights
        4. Record statistics, i.e loss, accuracy and time per epoch
        """

        # To indicate model in training phase
        # Will also turn on dropout in case we use it
        self.model.train()

        epoch_loss = 0
        epoch_acc = 0

        with tqdm(self.trainingDataLoader) as tqdmObject:

            startTime = time.time()
            for (x, y) in tqdmObject:
                tqdmObject.set_description(desc=f"Epoch {epoch+1}")
                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                # Step 1
                y_pred = self.model(x)

                # Step 2
                loss = self.criterion(y_pred, y)
                acc = self.__calculateAccuracy(y_pred, y)

                # Step 3
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                tqdmObject.set_postfix(accuracy=epoch_acc/len(self.trainingDataLoader),
                                       loss=epoch_loss/len(self.trainingDataLoader))
            # if self.schedule:
            #     self.schedule.step()
            endTime = time.time()

        trainingMinutes, trainingSeconds = self.__getTime(startTime, endTime)
        trainingSeconds += 60 * trainingMinutes
        return epoch_acc/len(self.trainingDataLoader), epoch_loss/len(self.trainingDataLoader), trainingSeconds

    def _evaluateModel(self):
        """
        Abstraction layer for validation steps
        1. Make predictions
        2. Calculate loss, accuracy
        3. Record statistics, i.e loss, accuracy and time per epoch
        """

        # To indicate model in training phase
        # Will also turn on dropout in case we use it
        self.model.eval()

        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():
            for (x, y) in self.testingDataLoader:

                x = x.to(self.device)
                y = y.to(self.device)

                # Step 1
                y_pred = self.model(x)

                # Step 2
                loss = self.criterion(y_pred, y)
                acc = self.__calculateAccuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_acc/len(self.testingDataLoader), epoch_loss/len(self.testingDataLoader)

    def trainEpochs(self, epochs=10, plot_results=True):
        """
        Will build more control into it, for now keeping it limited to 

        epochs: how many epochs to train for
        validate: do we even have a validation dataset
        plot_results: whether to plot training, testing and validation losses, accuracies 
        """

        for epoch in range(epochs):
            startTime = time.time()
            trainingAccuracy, trainingLoss, trainingSeconds = self._trainModel(
                epoch)
            testingAccuracy, testingLoss = self._evaluateModel()
            endTime = time.time()
            self.schedule.step(trainingAccuracy)
            epochMinutes, epochSeconds = self.__getTime(startTime, endTime)
            epochSeconds += epochMinutes*60
            print("TrainingLoss:%.2f|TrainingAccuracy:%.2f|EpochTime:%.2fs|TestingLoss:%.2f|TestingAccuracy:%.2f\n" % (
                trainingLoss, trainingAccuracy*100, epochSeconds, testingLoss, testingAccuracy*100))

            self.trainingHistory["loss"].append(trainingLoss)
            self.trainingHistory["accuracy"].append(trainingAccuracy)

            self.testingHistory["loss"].append(testingLoss)
            self.testingHistory["accuracy"].append(testingAccuracy)

        if plot_results:

            X = np.arange(1, epochs+1)

            plt.figure(1)
            plt.plot(X, self.trainingHistory['loss'], label='train_loss')
            plt.plot(X, self.testingHistory['loss'], label='test_loss')
            plt.legend()
            plt.savefig("./outputs/" + self.modelName + "LossVsEpochs.jpg")

            plt.figure(2)
            plt.plot(
                X, self.trainingHistory['accuracy'], label='train_acc')
            plt.plot(
                X, self.testingHistory['accuracy'], label='test_acc')
            plt.legend()
            plt.savefig("./outputs/" + self.modelName + "AccVsEpochs.jpg")
