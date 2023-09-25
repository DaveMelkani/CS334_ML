import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix

bin_train_data = pd.read_csv('binary_train.csv', header=None, dtype=int)
bin_train_data = pd.read_csv('binary_train.csv', header=None, dtype=int, error_bad_lines=False)

cnt_train_data = pd.read_csv('count_train.csv', header=None, dtype=int)
cnt_train_data = pd.read_csv('count_train.csv', header=None, dtype=int, error_bad_lines=False)

bin_train_labels = bin_train_data.iloc[:, 0]
bin_train_features = bin_train_data.iloc[:, 1:]

cnt_train_labels = cnt_train_data.iloc[:, 0]
cnt_train_features = cnt_train_data.iloc[:, 1:]

# Train the classifier
clf_1 = MultinomialNB()
clf_1.fit(bin_train_features, bin_train_labels)

clf_2 = BernoulliNB()
clf_2.fit(cnt_train_features, cnt_train_labels)

bin_test_data = pd.read_csv('binary_test.csv', header=None, dtype=int)
cnt_test_data = pd.read_csv('count_test.csv', header=None, dtype=int)

bin_test_labels = bin_test_data.iloc[:, 0]
bin_test_features = bin_test_data.iloc[:, 1:]

cnt_test_labels = cnt_test_data.iloc[:, 0]
cnt_test_features = cnt_test_data.iloc[:, 1:]

bin_predictions = clf_1.predict(bin_test_features)
cnt_predictions = clf_2.predict(cnt_test_features)

# Calculate the confusion matrix
bin_cm = confusion_matrix(bin_test_labels, bin_predictions)
cnt_cm = confusion_matrix(cnt_test_labels, cnt_predictions)
print('Binary Confusion matrix:\n', bin_cm)
print('Count Confusion matrix:\n', cnt_cm)

# Calculate the number of mistakes
bin_num_mistakes = bin_cm[0][1] + bin_cm[1][0]
cnt_num_mistakes = cnt_cm[0][1] + cnt_cm[1][0]
print('Number of binary mistakes:', bin_num_mistakes)
print('Number of count mistakes:', cnt_num_mistakes)
