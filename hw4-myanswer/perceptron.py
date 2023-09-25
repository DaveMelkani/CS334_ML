import argparse
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch
        self.w = None

    def train(self, xFeat, y, xVal=None, yVal=None):
        stats_train = {}
        stats_val = {}
        n, d = xFeat.shape
        self.w = np.zeros(d)
        mistakes_train = -1
        mistakes_val = -1
        epoch = 0

        while mistakes_train != 0 and epoch < self.mEpoch:
            mistakes_train = 0
            for i in range(n):
                xi = xFeat[i]
                yi = y[i]
                yp = self.predict(xi)
                if yp * yi <= 0:
                    mistakes_train += 1
                    self.w += yi * xi
            epoch += 1
            stats_train[epoch] = mistakes_train
            if xVal is not None and yVal is not None:
                mistakes_val = self.evaluate(xVal, yVal)
                stats_val[epoch] = mistakes_val
        return stats_train, stats_val

    def predict(self, xFeat):
        yPred = np.dot(xFeat, self.w)
        yHat = np.where(yPred > 0, 1, -1)
        return yHat

    def evaluate(self, xFeat, yTrue):
        yHat = self.predict(xFeat)
        mistakes = np.sum(yHat != yTrue)
        return mistakes

def k_fold_cv(X, y, k, epochs):
    kf = KFold(n_splits=k, shuffle=True)
    train_stats = {}
    val_stats = {}

    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f'Fold {i+1}/{k}')
        xTrain, yTrain = X[train_idx], y[train_idx]
        xVal, yVal = X[val_idx], y[val_idx]
        
        model = Perceptron(epochs)
        train_stats_fold, val_stats_fold = model.train(xTrain, yTrain, xVal, yVal)
        
        for epoch, mistakes_train in train_stats_fold.items():
            train_stats.setdefault(epoch, 0)
            train_stats[epoch] += mistakes_train
        
        for epoch, mistakes_val in val_stats_fold.items():
            val_stats.setdefault(epoch, 0)
            val_stats[epoch] += mistakes_val

    for epoch, mistakes_train in train_stats.items():
        train_stats[epoch] /= k
    
    for epoch, mistakes_val in val_stats.items():
        val_stats[epoch] /= k
    
    return train_stats, val_stats

def calc_mistakes(yHat, yTrue):
    err = np.sum(yHat != yTrue)
    return err

def top_words(weights, vocabulary, n=15):
    word_weights = list(zip(vocabulary, weights))
    word_weights.sort(key=lambda x: x[1])
    neg_words = [word for word, weight in word_weights[:n]]
    pos_words = [word for word, weight in word_weights[-n:][::-1]]
    return pos_words, neg_words

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    args = parser.parse_args()

    # read in the data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain).reshape(-1)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest).reshape(-1)

    # perform k-fold cross-validation
    k = 5
    train_stats, val_stats = k_fold_cv(xTrain, yTrain, k, args.epoch)

    # print the training and validation mistakes for each epoch
    print("Epoch\tTrain Mistakes\tValidation Mistakes")
    for epoch in range(1, args.epoch+1):
        train_mistakes = train_stats[epoch]
        val_mistakes = val_stats[epoch]
        print(f"{epoch}\t{train_mistakes:.2f}\t{val_mistakes:.2f}")





if __name__ == "__main__":
    main()