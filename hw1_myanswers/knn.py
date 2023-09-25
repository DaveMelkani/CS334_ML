import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


class Knn(object):
    k = 0    # number of neighbors to use
    xTrain = None
    yTrain = None

    def __init__(self, xTrain, yTrain, k):
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.k = k


    def train(self, xFeat, y):
        self.xFeat = xFeat
        self.y = y
        return self

    def predict(self, xFeat):
        yHat = []
        for sample in xFeat:
            distances = [np.linalg.norm(np.array(list(map(float, sample))) - np.array(list(map(float, train_sample)))) for train_sample in self.xTrain]
            closest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
            labels = [self.yTrain[i] for i in closest_indices]
            label_counts = Counter(labels)
            if len(label_counts) == 0:
                yHat.append("No nearest neighbors found")
            else:
                yHat.append(label_counts.most_common(1)[0][0])
        return yHat




def accuracy(yHat, yTrue):
    num_correct = sum(yHat == yTrue)
    acc = num_correct / len(yTrue)
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)

    k_values = range(1, 11)
    train_accs = []
    test_accs = []
    for k in k_values:
        knn = Knn(xTrain= xTrain, yTrain=[1, 2, 3], k=3)
        knn.train(xTrain, yTrain['label'])
        # predict the training dataset
        yHatTrain = knn.predict(xTrain)
        trainAcc = accuracy(yHatTrain, yTrain['label'])
        train_accs.append(trainAcc)
        # predict the test dataset
        yHatTest = knn.predict(xTest)
        testAcc = accuracy(yHatTest, yTest['label'])
        test_accs.append(testAcc)
    results = pd.DataFrame({'k': k_values, 'trainAcc': train_accs, 'testAcc': test_accs})
    results.plot(x='k', y=['trainAcc', 'testAcc'], kind='line')
    plt.show()
    '''
    The computational complexity of the predict function can be expressed in terms of the 
    training size (n), the number of features (d), and the number of neighbors (k). In the predict function, 
    the main operation is the calculation of the distances between the input data point and all 
    the data points in the training set. The time complexity of this operation is O(n * d), as we 
    need to calculate the distances between the input data point and all n training points, and each 
    distance calculation involves d features. Next, we need to sort the distances in ascending order, 
    which has a time complexity of O(n * log(n)). Then, we select the k nearest neighbors, 
    which has a time complexity of O(k). Therefore, the overall time complexity of 
    the predict function is O(n * d + n * log(n) + k). It's worth noting that the value of k has a 
    much smaller impact on the overall time complexity compared to n and d. As k is typically a small number, 
    it can be considered as a constant value in the time complexity analysis, and the overall time complexity 
    can be simplified to O(n * d + n * log(n)).
    '''

if __name__ == "__main__":
    main()

