import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def extract_features(df):
     # Convert date column to datetime object
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    
    # Extract hour, day of the week, and day of the year
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Drop the original date column
    df = df.drop(columns=['date'])
    
    return df


def select_features(df):
    # # Select the columns we want to keep
    cols_to_keep = ['lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint']
    return df[cols_to_keep]


def preprocess_data(trainDF, testDF):
    # Remove the 'date' column from both train and test data

    # Normalize the features using z-score normalization
    trainDF.iloc[:, :-1] = (trainDF.iloc[:, :-1] - trainDF.iloc[:, :-1].mean()) / trainDF.iloc[:, :-1].std()
    testDF.iloc[:, :-1] = (testDF.iloc[:, :-1] - testDF.iloc[:, :-1].mean()) / testDF.iloc[:, :-1].std()

    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)

    #part(b)
    # Calculate the correlation matrix
    corr_matrix = xTrainTr.corr()
    print(corr_matrix)

    # Plot the correlation matrix as a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()


if __name__ == "__main__":
    main()
