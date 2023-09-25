import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load the dataset
x_train = pd.read_csv('q4xTrain.csv')
y_train = pd.read_csv('q4yTrain.csv')
x_test = pd.read_csv('q4xTest.csv')
y_test = pd.read_csv('q4yTest.csv')

# set up k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# set up parameter grids for grid search
knn_params = {'n_neighbors': range(1, 21)}
tree_params = {'max_depth': range(1, 21)}

# perform grid search with cross-validation for k-nn
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=kf)
knn_grid.fit(x_train, y_train.values.ravel())

# perform grid search with cross-validation for decision tree
tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=kf)
tree_grid.fit(x_train, y_train.values.ravel())

# get the optimal hyperparameters for each model
knn_opt = knn_grid.best_params_['n_neighbors']
tree_opt = tree_grid.best_params_['max_depth']

# train the models with the optimal hyperparameters
knn = KNeighborsClassifier(n_neighbors=knn_opt)
tree = DecisionTreeClassifier(max_depth=tree_opt)

knn.fit(x_train, y_train.values.ravel())
tree.fit(x_train, y_train.values.ravel())

# evaluate the models on the test set
knn_pred = knn.predict(x_test)
tree_pred = tree.predict(x_test)

knn_acc = accuracy_score(y_test, knn_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print(f"K-NN optimal parameter k: {knn_opt}")
print(f"Decision Tree optimal parameter max_depth: {tree_opt}")
print(f"K-NN accuracy on test set: {knn_acc:.3f}")
print(f"Decision Tree accuracy on test set: {tree_acc:.3f}")
