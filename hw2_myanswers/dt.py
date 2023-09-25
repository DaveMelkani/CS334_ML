import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
from scipy.stats import entropy
import matplotlib.pyplot as plt


class Node:
    def __init__(self, gain=None, value=None, feature=None, threshold=None, data_left=None, data_right=None, splitVal=None):
        self.gain = gain
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = data_left
        self.right = data_right
        self.splitVal=splitVal

class DecisionTree(object):
    maxDepth = 0     
    minLeafSample = 0 
    criterion = None   


    def __init__(self, criterion, maxDepth, minLeafSample):
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.root=None

    def mode(self, y):
        counts = Counter(y)
        mode, modeCount = counts.most_common(1)[0]
        return mode

    def impurity(self, y):
        y_array = np.array(y)
        if self.criterion == 'entropy':
            unique_values, value_counts = np.unique(y_array, return_counts=True)
            entropy_val = entropy(value_counts, base=2)
            return entropy_val
        
        elif self.criterion == 'gini':
            unique_values, value_counts = np.unique(y_array, return_counts=True)
            probabilities = value_counts / y_array.size
            gini_impurity = 1 - np.sum(probabilities**2)
            return gini_impurity
        
    def train(self, xFeat, y):
        self.root= Node(value=self.mode(y))
        yVals = np.array(y)
        imp= float('inf')
        if not self.impurity(y) or len(xFeat) < 2*self.minLeafSample or not self.maxDepth:
            return self
        for f in xFeat:
            features = np.sort(xFeat[f])
            n1=features.size-self.minLeafSample+1
            n2=self.minLeafSample-1
            for i in range(n2, n1):
                Lefts=xFeat[f] <= features[i]
                Rights=xFeat[f] > features[i]
                yLeft = yVals[Lefts]
                yRight = yVals[Rights]
                LeftWeight = yLeft.size/yVals.size
                RightWeight = yRight.size/yVals.size
                weightedImpLeft= LeftWeight * self.impurity(yLeft)
                weightedImpRight= RightWeight * self.impurity(yRight)
                splitImp = weightedImpLeft + weightedImpRight
                if splitImp < imp:
                    self.root.splitVal = features[i]
                    self.root.feature = f
                    imp = splitImp
        LeftVals=xFeat[self.root.feature] <= self.root.splitVal
        RightVals=xFeat[self.root.feature] > self.root.splitVal
        xFeatL = xFeat[LeftVals]
        yLeft = yVals[LeftVals]
        xFeatR = xFeat[RightVals]
        yRight = yVals[RightVals]
        newTreeLetf=DecisionTree(self.criterion, self.maxDepth-1, self.minLeafSample)
        newTreeRight=DecisionTree(self.criterion, self.maxDepth-1, self.minLeafSample)
        self.root.left = newTreeLetf.train(xFeatL, yLeft)
        self.root.right = newTreeRight.train(xFeatR, yRight)
        return self


    def predict(self, xFeat):
        yHat = []
        n=len(xFeat)
        for i in range(n):
            pred = self.predictRecurse(xFeat.iloc[i])
            yHat.append(pred)
        return yHat


    def predictRecurse(self, x):
        node = self.root
        if node.left is None and node.right is None:
            return node.value
        elif x[node.feature] > node.splitVal:
            return node.right.predictRecurse(x)
        else:
            return node.left.predictRecurse(x)


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    dt.train(xTrain, yTrain['label'])
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
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
    parser.add_argument("--xTrain",                        default="q4xTrain.csv",
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


    maxdepth_values = range(2, 15)
    min_samples_leaf = 4
    trainAccuracies = []
    testAccuracies = []
    for maxdepth in maxdepth_values:
        dt = DecisionTree('gini', maxdepth, min_samples_leaf)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        trainAccuracies.append(trainAcc)
        testAccuracies.append(testAcc)
    fig, ax = plt.subplots()
    ax.plot(maxdepth_values, trainAccuracies, label="Training Acc")
    ax.plot(maxdepth_values, testAccuracies, label="Test Acc")
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Acc')
    ax.set_title('Accuracy as a relation to Maximum Depth')
    ax.legend()
    plt.show()

    # plot 2
    min_samples_leaf_values1 = range(2, 15)
    maxdepth1 = 7
    trainAccuracies1 = []
    testAccuracies1 = []
    for min_samples_leaf in min_samples_leaf_values1:
        dt = DecisionTree('gini', maxdepth1, min_samples_leaf)
        trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
        trainAccuracies1.append(trainAcc)
        testAccuracies1.append(testAcc)
    fig, ax = plt.subplots()
    ax.plot(min_samples_leaf_values1, trainAccuracies1, label="Training Acc")
    ax.plot(min_samples_leaf_values1, testAccuracies1, label="Test Acc")
    ax.set_xlabel('Min Num Samples in Leaf')
    ax.set_ylabel('Acc')
    ax.set_title('Acc in Relation to Min Num Samples in Leaf')
    ax.legend()
    plt.show()



if __name__ == "__main__":
    main()
