import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time

 
def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set. 

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    # Split the dataset into train and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(xFeat, y, test_size=testSize)
    
    # Train the model and measure training time
    start = time.time()
    model.fit(xTrain, yTrain)
    trainTime = time.time() - start
    
    # Calculate AUC on training set
    yTrainProb = model.predict_proba(xTrain)[:,1]
    trainAuc = roc_auc_score(yTrain, yTrainProb)
    
    # Calculate AUC on test set and measure testing time
    start = time.time()
    yTestProb = model.predict_proba(xTest)[:,1]
    testAuc = roc_auc_score(yTest, yTestProb)
    testTime = time.time() - start
    
    return trainAuc, testAuc, trainTime + testTime


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set. 


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    kf = KFold(n_splits=k)
    train_auc_scores = []
    test_auc_scores = []
    start_time = time.time()
    
    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat.iloc[train_index], xFeat.iloc[test_index]
        yTrain, yTest = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(xTrain, yTrain)
        yTrainPred = model.predict_proba(xTrain)[:, 1]
        yTestPred = model.predict_proba(xTest)[:, 1]
        
        train_auc_scores.append(roc_auc_score(yTrain, yTrainPred))
        test_auc_scores.append(roc_auc_score(yTest, yTestPred))
    
    trainAuc = np.mean(train_auc_scores)
    testAuc = np.mean(test_auc_scores)
    timeElapsed = time.time() - start_time
    
    return trainAuc, testAuc, timeElapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : nd-array with shape n x d
        Features of the dataset 
    y : 1-array with shape n x 1
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout. 

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    trainAuc = []
    testAuc = []
    start_time = time.time()
    
    for i in range(s):
        # split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(xFeat, y, test_size=testSize)
        
        # fit model on training set
        model.fit(x_train, y_train)
        
        # predict probabilities on training and testing sets
        y_train_prob = model.predict_proba(x_train)[:,1]
        y_test_prob = model.predict_proba(x_test)[:,1]
        
        # compute AUC for training and testing sets
        train_auc = roc_auc_score(y_train, y_train_prob)
        test_auc = roc_auc_score(y_test, y_test_prob)
        
        trainAuc.append(train_auc)
        testAuc.append(test_auc)
    
    timeElapsed = time.time() - start_time
    
    # compute average AUC across all splits
    avg_train_auc = np.mean(trainAuc)
    avg_test_auc = np.mean(testAuc)
    
    return avg_train_auc, avg_test_auc, timeElapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier 
    xTrain : nd-array with shape nxd
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data
    xTest : nd-array with shape mxd
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain['label'],
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest['label'],
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.70)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.70, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.70, 10)
    # train it using all the data and assess the true value
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                           columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
