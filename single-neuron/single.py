import numpy as np
X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 75],
    [6, 80],
    [7, 85],
    [8, 90],
    [9, 95]
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

X = (X - X.mean(axis=0)) / X.std(axis=0)
np.random.seed(42)

W = np.random.randn(2, 1)
b = np.zeros((1, 1))

learning_rate = 0.1
epochs = 5000


#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):

    #foward propagation
    Z = np.dot(X, W) + b
    A = sigmoid(Z)

    epsilon = 1e-10
    loss = -np.mean(
        y * np.log(A + epsilon)
        +
        (1 - y) * np.log(1 - A + epsilon)
    )

    #backpropagation
    m = X.shape[0]
    dZ = A - y
    dW = (1 / m) * np.dot(X.T, dZ)
    db = (1 / m) * np.sum(dZ)

    W -= learning_rate * dW
    b -= learning_rate * db


    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}  Loss = {loss:.6f}")
        
print("\nTraining Finished!")
print("Weights:")
print(W)

print("\nBias:")
print(b)

print("\nPredictions")

predictions = sigmoid(np.dot(X, W) + b)

for i in range(len(predictions)):

    label = 1 if predictions[i] >= 0.5 else 0

    print(
        f"Actual: {int(y[i][0])} | "
        f"Probability: {predictions[i][0]:.4f} | "
        f"Predicted: {label}"
    )

print("\nTest New Student")

study_hours = 7
attendance = 88
new_student = np.array([[study_hours, attendance]])

new_student = (new_student - X.mean(axis=0)) / X.std(axis=0)
probability = sigmoid(np.dot(new_student, W) + b)
prediction = 1 if probability >= 0.5 else 0

print(f"Study Hours : {study_hours}")
print(f"Attendance  : {attendance}")
print(f"Probability : {probability[0][0]:.4f}")
print(f"Prediction  : {'PASS' if prediction else 'FAIL'}")
