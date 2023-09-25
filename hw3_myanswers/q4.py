import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate some data for testing the linear regression algorithm
X, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=1)

# Define the batch sizes to test
batch_sizes = [1, 2, 5, 10, 20, 50, 100, len(X)]

# Define a function to calculate the mean squared error of the linear regression model
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Define a function to train the linear regression model using stochastic gradient descent
def train_sgd(X, y, batch_size, learning_rate, num_epochs):
    # Initialize the weights randomly
    np.random.seed(1)
    w = np.random.randn(X.shape[1])
    b = np.random.randn()
    # Initialize the loss history
    loss_history = []
    # Iterate over the epochs
    for epoch in range(num_epochs):
        # Shuffle the data randomly
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        # Iterate over the batches
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            # Compute the gradients
            y_pred = X_batch.dot(w) + b
            error = y_pred - y_batch
            grad_w = 2 * X_batch.T.dot(error) / len(X_batch)
            grad_b = 2 * np.mean(error)
            # Update the weights
            w -= learning_rate * grad_w
            b -= learning_rate * grad_b
        # Compute the loss on the full dataset
        y_pred = X.dot(w) + b
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)
    return w, b, loss_history

# Train the linear regression model using stochastic gradient descent for each batch size
learning_rate = 0.01
num_epochs = 100
results = []
for batch_size in batch_sizes:
    start_time = time.time()
    w, b, loss_history = train_sgd(X, y, batch_size, learning_rate, num_epochs)
    elapsed_time = time.time() - start_time
    # Evaluate the model on the training data
    y_pred_train = X.dot(w) + b
    mse_train = mean_squared_error(y, y_pred_train)
    # Evaluate the model on the test data
    X_test, y_test = make_regression(n_samples=500, n_features=10, noise=0.5, random_state=2)
    y_pred_test = X_test.dot(w) + b
    mse_test = mean_squared_error(y_test, y_pred_test)
    # Save the results
    results.append({'batch_size': batch_size, 'mse_train': mse_train, 'mse_test': mse_test, 'time': elapsed_time})

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Plot the training data MSE vs. time for different batch sizes
for result in results:
    batch_size = result['batch_size']
    mse_train = result['mse_train']
    time = result['time']
    ax[0].plot(time, mse_train, 'o-', label=f'Batch size {batch_size}')

ax[0].set_title('Mean Squared Error vs. Time (Training Data)')
ax[0].set_xlabel('Time (seconds)')
ax[0].set_ylabel('Mean Squared Error')
ax[0].legend()

# Plot the test data MSE vs. time for different batch sizes
for result in results:
    batch_size = result['batch_size']
    mse_test = result['mse_test']
    time = result['time']
    ax[1].plot(time, mse_test, 'o-', label=f'Batch size {batch_size}')
ax[1].set_title('Mean Squared Error vs. Time (Test Data)')
ax[1].set_xlabel('Time (seconds)')
ax[1].set_ylabel('Mean Squared Error')
ax[1].legend()

# Add the point for the closed form solution
w_cf = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
b_cf = np.mean(y)
mse_train_cf = mean_squared_error(y, X.dot(w_cf) + b_cf)
mse_test_cf = mean_squared_error(y_test, X_test.dot(w_cf) + b_cf)
ax[0].plot(0, mse_train_cf, 's', markersize=10, label='Closed form')
ax[1].plot(0, mse_test_cf, 's', markersize=10, label='Closed form')

plt.tight_layout()
plt.show()



# b)

'''
- Larger batch sizes result in slower convergence but with less noise in weight updates.
- Smaller batch sizes result in faster convergence but with more noise in weight updates.
- Increasing batch size decreases training time but increases mean squared error (MSE) on both training and test data.
- Closed-form solution has the lowest MSE on both training and test data but takes longer to compute.
- There is a trade-off between computational efficiency and accuracy of closed-form solution.
- Small batch sizes may be preferred for smaller datasets or when faster convergence is desired.
- Large batch sizes may be preferred for larger datasets or when more stable convergence is desired.
'''