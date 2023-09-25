import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing



def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    data = pd.read_csv(filename, header=None)
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test


def build_vocab_map(train_data):
    vocab_map = {}
    num_emails = len(train_data)
    for i in range(num_emails):
        words = set(train_data.iloc[i, 0].split())
        for word in words:
            if word in vocab_map:
                vocab_map[word] += 1
            else:
                vocab_map[word] = 1
    vocab_list = [word for word in vocab_map if vocab_map[word] >= 30]
    return vocab_list


def construct_binary(train_data, test_data, vocab_list):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    num_train_emails = len(train_data)
    num_test_emails = len(test_data)
    num_vocab = len(vocab_list)
    
    train_set = np.zeros((num_train_emails, num_vocab), dtype=int)
    test_set = np.zeros((num_test_emails, num_vocab), dtype=int)
    
    # convert training set to binary representation
    for i in range(num_train_emails):
        words = train_data.iloc[i, 0].split()
        for j in range(num_vocab):
            if vocab_list[j] in words:
                train_set[i, j] = 1
    
    # convert test set to binary representation
    for i in range(num_test_emails):
        words = test_data.iloc[i, 0].split()
        for j in range(num_vocab):
            if vocab_list[j] in words:
                test_set[i, j] = 1
    
    return train_set, test_set



def construct_count(train_data, test_data, vocab_list):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    x_i, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise.
    """
    num_train_emails = len(train_data)
    num_test_emails = len(test_data)
    num_vocab = len(vocab_list)
    
    train_set = np.zeros((num_train_emails, num_vocab), dtype=int)
    test_set = np.zeros((num_test_emails, num_vocab), dtype=int)
    
    # convert training set to count representation
    for i in range(num_train_emails):
        words = train_data.iloc[i, 0].split()
        for j in range(num_vocab):
            train_set[i, j] = words.count(vocab_list[j])
    
    # convert test set to count representation
    for i in range(num_test_emails):
        words = test_data.iloc[i, 0].split()
        for j in range(num_vocab):
            test_set[i, j] = words.count(vocab_list[j])
    
    return train_set, test_set



def main():
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--data', type=str, required=True,
                        help='input dataset filename')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='directory to store output datasets')
    args = parser.parse_args()
    # Load the dataset and split it into training and test sets
    train, test = model_assessment(args.data)
    # Build vocabulary map
    vocab_map = build_vocab_map(train)
    # Construct binary datasets
    binary_train, binary_test = construct_binary(train, test, vocab_map)
    binary_train = pd.DataFrame(binary_train, columns=vocab_map)
    binary_test = pd.DataFrame(binary_test, columns=vocab_map)
    # Construct count datasets
    count_train, count_test = construct_count(train, test, vocab_map)
    count_train = pd.DataFrame(count_train, columns=vocab_map)
    count_test = pd.DataFrame(count_test, columns=vocab_map)
    # Output datasets to CSV files
    binary_train.to_csv(os.path.join(args.output_dir, 'binary_train.csv'), index=False)
    binary_test.to_csv(os.path.join(args.output_dir, 'binary_test.csv'), index=False)
    count_train.to_csv(os.path.join(args.output_dir, 'count_train.csv'), index=False)
    count_test.to_csv(os.path.join(args.output_dir, 'count_test.csv'), index=False)

    # print("Vocabulary size:", len(vocab_list))
    # print("Sample words:", vocab_list[:10])
    # print("Shape of binary training set:", binary_train_set.shape)
    # print("Shape of binary test set:", binary_test_set.shape)
    # print("Shape of count training set:", count_train_set.shape)
    # print("Shape of count test set:", count_test_set.shape)






if __name__ == "__main__":
    main()
