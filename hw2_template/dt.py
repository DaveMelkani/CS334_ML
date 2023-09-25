import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.tree = self.build_tree(xFeat, y)
        return self


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        return yHat
    
    def build_tree(self, xFeat, y, depth=0):
        # Stopping criteria
        if (depth >= self.maxDepth) or (len(y) < self.minLeafSample) or (len(np.unique(y)) == 1):
            return np.bincount(y).argmax()
        # Find the best attribute to split on
        best_attr, best_gain = self.get_best_split(xFeat, y)
        # Create a new node with the best attribute
        node = {'attribute': best_attr, 'children': {}}
        # Recursively split the data and create branches
        for value in np.unique(xFeat[:, best_attr]):
            mask = xFeat[:, best_attr] == value
            node['children'][value] = self.build_tree(xFeat[mask], y[mask], depth+1)
        return node
    
    def get_best_split(self, xFeat, y):
        best_attr = None
        best_gain = -np.inf
        for attr in range(xFeat.shape[1]):
            gain = self.get_information_gain(xFeat[:, attr], y)
            if gain > best_gain:
                best_attr = attr
                best_gain = gain
        return best_attr, best_gain
    
    def get_information_gain(self, feature, y):
        # Calculate entropy before the split
        H_before = self.entropy(y)
        # Calculate entropy after the split
        H_after = 0
        for value in np.unique(feature):
            mask = feature == value
            H_after += np.sum(mask) / len(y) * self.entropy(y[mask])
        # Calculate information gain
        IG = H_before - H_after
        return IG
    
    def entropy(self, y):
        # Calculate entropy of a single node
        p1 = np.sum(y == 1) / len(y)
        p0 = 1 - p1
        if p0 == 0 or p1 == 0:
            return 0
        else:
            return -p0 * np.log2(p0) - p1 * np.log2(p1)


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain"                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
