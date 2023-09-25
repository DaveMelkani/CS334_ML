import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        # Add column of ones to xTrain and xTest for bias term
        n, d = xTrain.shape
        xTrain = np.hstack((np.ones((n, 1)), xTrain))
        m, _ = xTest.shape
        xTest = np.hstack((np.ones((m, 1)), xTest))

        # Calculate closed-form solution for beta coefficient vector 
        start_time = time.time()
        xTx = xTrain.T.dot(xTrain)
        xTy = xTrain.T.dot(yTrain)
        self.beta = np.linalg.solve(xTx, xTy)

        # Calculate mean squared errors for training and testing data after predicting y values
        train_mse = self.mse(xTrain, yTrain)
        test_mse = self.mse(xTest, yTest)

        elapsed_time = time.time() - start_time
        # Create dictionary with stats
        trainStats = {0: {'time': elapsed_time, 'train-mse': train_mse, 'test-mse': test_mse}}
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

    args = parser.parse_args()
    # load the train and test data
    # xTrain = file_to_numpy(args.xTrain)
    xTrain = file_to_numpy(args.xTrain)[:-1, :]
    yTrain = file_to_numpy(args.yTrain).flatten()
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    xTrain = np.genfromtxt('new_xTrain.csv', delimiter=',')
    print(xTrain.shape)  # should print (num_samples, num_features)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
