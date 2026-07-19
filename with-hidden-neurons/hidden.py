import numpy as np

X = np.array([
    [2,60],
    [3,65],
    [4,70],
    [5,75],
    [6,80],
    [7,85],
    [8,90],
    [9,95]
], dtype=np.float64)

y = np.array([
    [0],
    [0],
    [0],
    [0],
    [1],
    [1],
    [1],
    [1]
], dtype=np.float64)

mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

np.random.seed(42)
input_size = 2
hidden_size = 2
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1,hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.1
epochs = 5000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)

#training
for epoch in range(epochs):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    epsilon = 1e-10

    loss = -np.mean(
        y*np.log(A2 + epsilon)
        +
        (1-y)*np.log(1-A2 + epsilon)
    )

    #backprop
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (1/m) * np.dot(X.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}  Loss = {loss:.6f}")
