import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

binary_train_data = pd.read_csv("binary_train.csv", header=None)
binary_test_data = pd.read_csv("binary_test.csv", header=None)
count_train_data = pd.read_csv("count_train.csv", header=None)
count_test_data = pd.read_csv("count_test.csv", header=None)


binary_train_labels = binary_train_data.iloc[:, 0]
binary_train_features = binary_train_data.iloc[:, 1:]
binary_test_labels = binary_test_data.iloc[:, 0]
binary_test_features = binary_test_data.iloc[:, 1:]
count_train_labels = count_train_data.iloc[:, 0]
count_train_features = count_train_data.iloc[:, 1:]
count_test_labels = count_test_data.iloc[:, 0]
count_test_features = count_test_data.iloc[:, 1:]


binary_lr_model = LogisticRegression()
binary_lr_model.fit(binary_train_features, binary_train_labels)


binary_lr_pred = binary_lr_model.predict(binary_test_features)


binary_lr_accuracy = accuracy_score(binary_test_labels, binary_lr_pred)


count_lr_model = LogisticRegression()
count_lr_model.fit(count_train_features, count_train_labels)


count_lr_pred = count_lr_model.predict(count_test_features)


count_lr_accuracy = accuracy_score(count_test_labels, count_lr_pred)


print("Number of mistakes for binary datasets: ", len(binary_test_labels) - binary_lr_accuracy * len(binary_test_labels))
print("Number of mistakes for count datasets: ", len(count_test_labels) - count_lr_accuracy * len(count_test_labels))
