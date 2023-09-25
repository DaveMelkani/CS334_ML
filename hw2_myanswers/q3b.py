import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Load the datasets
x_train = pd.read_csv('q4xTrain.csv')
y_train = pd.read_csv('q4yTrain.csv')
x_test = pd.read_csv('q4xTest.csv')
y_test = pd.read_csv('q4yTest.csv')

# Train k-NN on the entire training dataset with optimal hyperparameter
knn_opt = KNeighborsClassifier(n_neighbors=7)
knn_opt.fit(x_train, y_train)

# Evaluate AUC and accuracy on the test set for the k-NN model with optimal hyperparameter
y_pred = knn_opt.predict_proba(x_test)[:, 1]
knn_opt_auc = roc_auc_score(y_test, y_pred)
knn_opt_acc = accuracy_score(y_test, knn_opt.predict(x_test))
print("K-NN model with optimal hyperparameter:")
print(f"AUC on test set: {knn_opt_auc:.3f}")
print(f"Accuracy on test set: {knn_opt_acc:.3f}")

# Create 3 datasets where you randomly remove 5%, 10%, and 20% of the original training data
np.random.seed(1)
sub1 = np.random.choice(x_train.index, size=int(0.05*len(x_train)), replace=False)
sub2 = np.random.choice(x_train.index, size=int(0.1*len(x_train)), replace=False)
sub3 = np.random.choice(x_train.index, size=int(0.2*len(x_train)), replace=False)

# Save the subset CSV files with label column
x_train.loc[sub1].join(y_train).to_csv('subset_1.csv', index=False)
x_train.loc[sub2].join(y_train).to_csv('subset_2.csv', index=False)
x_train.loc[sub3].join(y_train).to_csv('subset_3.csv', index=False)

# Train k-NN on each subset with optimal hyperparameter and evaluate AUC and accuracy on the test set
knn1 = KNeighborsClassifier(n_neighbors=7)
knn2 = KNeighborsClassifier(n_neighbors=7)
knn3 = KNeighborsClassifier(n_neighbors=7)

subset1 = pd.read_csv('subset_1.csv')
subset2 = pd.read_csv('subset_2.csv')
subset3 = pd.read_csv('subset_3.csv')

knn1.fit(subset1.drop(columns=['label']), subset1['label'])
knn2.fit(subset2.drop(columns=['label']), subset2['label'])
knn3.fit(subset3.drop(columns=['label']), subset3['label'])

knn1_auc = roc_auc_score(y_test, knn1.predict_proba(x_test)[:, 1])
knn1_acc = accuracy_score(y_test, knn1.predict(x_test))
knn2_auc = roc_auc_score(y_test, knn2.predict_proba(x_test)[:, 1])
knn2_acc = accuracy_score(y_test, knn2.predict(x_test))
knn3_auc = roc_auc_score(y_test, knn3.predict_proba(x_test)[:, 1])
knn3_acc = accuracy_score(y_test, knn3.predict(x_test))


# Print the results
print("K-NN model on subset 1 (5%):")
print(f"AUC on test set: {knn1_auc:.3f}")
print(f"Accuracy on test set: {knn1_acc:.3f}")
print("K-NN model on subset 2 (10%):")
print(f"AUC on test set: {knn2_auc:.3f}")
print(f"Accuracy on test set: {knn2_acc:.3f}")
print("K-NN model on subset 3 (20%):")
print(f"AUC on test set: {knn3_auc:.3f}")
print(f"Accuracy on test set: {knn3_acc:.3f}")
