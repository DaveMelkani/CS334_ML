import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        n, d = xTrain.shape
        m = self.bs  # mini-batch size

        # initialize weights
        self.weights = np.random.randn(d)

        # shuffle the data
        perm = np.random.permutation(n)
        xTrain = xTrain[perm]
        yTrain = yTrain[perm]

        # do SGD
        idx = 0
        for epoch in range(self.mEpoch):
            # iterate over mini-batches
            for batch in range(0, n, m):
                xBatch = xTrain[batch:batch+m]
                yBatch = yTrain[batch:batch+m]
                # compute gradient
                grad = np.dot(xBatch.T, np.dot(xBatch, self.weights) - yBatch) / m
                # update weights
                self.weights -= self.lr * grad
                # compute train and test error every 10 iterations
                if idx % 10 == 0:
                    trainMSE = np.mean((np.dot(xTrain, self.weights) - yTrain) ** 2)
                    testMSE = np.mean((np.dot(xTest, self.weights) - yTest) ** 2)
                    trainStats[idx] = {'time': time.time(), 'train-mse': trainMSE, 'test-mse': testMSE}
                idx += 1

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    # print("hi")
    # print(os.getcwd())
    # print("bye")
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)

    # set batch size to 1 and maximum epoch to 1000
    bs = 1
    mEpoch = 1000

    # train and test model with optimal learning rate from part (b)
    lr = 0.0001
    model = SgdLR(lr, bs, mEpoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

    # plot mean squared error on training data and test data as a function of epoch
    trainMSEs = [trainStats[idx]['train-mse'] for idx in trainStats]
    testMSEs = [trainStats[idx]['test-mse'] for idx in trainStats]
    epochs = [idx for idx in trainStats]

    fig, ax = plt.subplots()
    ax.plot(epochs, trainMSEs, label='Train MSE')
    ax.plot(epochs, testMSEs, label='Test MSE')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error')
    plt.show()


if __name__ == "__main__":
    main()

