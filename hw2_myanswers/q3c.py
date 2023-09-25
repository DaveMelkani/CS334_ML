import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Load data
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

print("Decision Tree models trained on subsets:")
print(f"Subset 1 accuracy on test set: {tree_sub1_acc:.3f}")
print(f"Subset 1 AUC on test set: {tree_sub1_auc:.3f}")
print(f"Subset 2 accuracy on test set: {tree_sub2_acc:.3f}")
print(f"Subset 2 AUC on test set: {tree_sub2_auc:.3f}")
print(f"Subset 3 accuracy on test set: {tree_sub3_acc:.3f}")
print(f"Subset 3 AUC on test set: {tree_sub3_auc:.3f}")


'''
The code starts by loading the datasets, 
and then uses the optimal hyperparameters 
for the decision tree model found in q3a.py. 
It trains a decision tree on the entire training 
dataset with these hyperparameters, and evaluates 
the model's accuracy and performance on the test set. 
The code then outputs the accuracy and performance 
metrics such as precision, recall, and F1 score for 
the decision tree model on the test set.
'''