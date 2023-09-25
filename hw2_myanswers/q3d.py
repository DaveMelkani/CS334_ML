
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from tabulate import tabulate

# Load Data
x_train = pd.read_csv('q4xTrain.csv')
y_train = pd.read_csv('q4yTrain.csv')
x_test = pd.read_csv('q4xTest.csv')
y_test = pd.read_csv('q4yTest.csv')

# get the optimal hyperparameters for the decision tree model from q3a.py
tree_opt = 8

# train a decision tree on the entire training dataset with optimal hyperparameters
tree_opt_model = DecisionTreeClassifier(max_depth=tree_opt)
tree_opt_model.fit(x_train, y_train)

# evaluate the decision tree model on the test set
tree_opt_pred = tree_opt_model.predict(x_test)

tree_opt_acc = accuracy_score(y_test, tree_opt_pred)
tree_opt_auc = roc_auc_score(y_test, tree_opt_model.predict_proba(x_test)[:, 1])

print(f"Optimal Decision Tree parameter max_depth: {tree_opt}")
print(f"Decision Tree accuracy on test set: {tree_opt_acc:.3f}")
print(f"Decision Tree AUC on test set: {tree_opt_auc:.3f}")

# use the subsets from q3b.py to train a new decision tree with optimal hyperparameters and evaluate on test set
subset1 = pd.read_csv('subset_1.csv')
subset2 = pd.read_csv('subset_2.csv')
subset3 = pd.read_csv('subset_3.csv')

tree_sub1_model = DecisionTreeClassifier(max_depth=tree_opt)
tree_sub2_model = DecisionTreeClassifier(max_depth=tree_opt)
tree_sub3_model = DecisionTreeClassifier(max_depth=tree_opt)

tree_sub1_model.fit(subset1.drop(columns=['label']), subset1['label'])
tree_sub2_model.fit(subset2.drop(columns=['label']), subset2['label'])
tree_sub3_model.fit(subset3.drop(columns=['label']), subset3['label'])

tree_sub1_pred = tree_sub1_model.predict(x_test)
tree_sub2_pred = tree_sub2_model.predict(x_test)
tree_sub3_pred = tree_sub3_model.predict(x_test)

tree_sub1_acc = accuracy_score(y_test, tree_sub1_pred)
tree_sub2_acc = accuracy_score(y_test, tree_sub2_pred)
tree_sub3_acc = accuracy_score(y_test, tree_sub3_pred)

tree_sub1_auc = roc_auc_score(y_test, tree_sub1_model.predict_proba(x_test)[:, 1])
tree_sub2_auc = roc_auc_score(y_test, tree_sub2_model.predict_proba(x_test)[:, 1])
tree_sub3_auc = roc_auc_score(y_test, tree_sub3_model.predict_proba(x_test)[:, 1])

# create a table of results
table = [["Optimal", tree_opt_acc, tree_opt_auc],
["Subset 1", tree_sub1_acc, tree_sub1_auc],
["Subset 2", tree_sub2_acc, tree_sub2_auc],
["Subset 3", tree_sub3_acc, tree_sub3_auc]]

print(tabulate(table, headers=["Model", "Accuracy", "AUC"]))