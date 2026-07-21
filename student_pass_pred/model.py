import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("student_nn.csv")

numerical = [
    "StudyHours",
    "Attendance",
    "SleepHours",
    "AssignmentsCompleted",
    "PreviousGrade"
]

for col in numerical:
    df[col] = df[col].fillna(df[col].mean())

df["InternetQuality"] = df["InternetQuality"].fillna(df["InternetQuality"].mode()[0])

df = pd.get_dummies(
    df,
    columns=["InternetQuality"],
    dtype=int
)

X = df.drop("Pass", axis=1)
y = df["Pass"].values.reshape(-1,1)

scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

np.random.seed(42)

input_size = X_train.shape[1]
hidden_size = 8
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))
learning_rate = 0.05
epochs = 5000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def relu(z):
    return np.maximum(0, z)
def relu_derivative(z):
    return ( z > 0 ).astype(float)


for epoch in range(epochs):

    #forward
    Z1 = np.dot(X_train,W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    #loss
    epsilon = 1e-10

    loss = -np.mean(
        y_train * np.log(A2 + epsilon)
        + 
        (1 - y_train) * np.log(1 - A2 + epsilon)
    )

    #backprop
    m = X_train.shape[0]
    dZ2 = A2 - y_train
    dW2 = (1/m) * np.dot(A1.T, dZ2)
    db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * relu_derivative(Z1) 
    dW1 = (1/m) * np.dot(X_train.T, dZ1)
    db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

    #update
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate *  dW2
    b2 -= learning_rate * db2

    print(f"Epoch {epoch} Loss={loss:.4f}")
    
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1,W2) + b2
A2 = sigmoid(Z2)

pred = (A2 >= 0.5).astype(int)
accuracy = np.mean(pred == y_test)
print()
print("Accuracy:", accuracy)
print()
for actual,predicted,prob in zip(y_test, pred, A2):
    print(
        f"Actual={actual[0]} "
        f"Predicted={predicted[0]} "
        f"Probability={prob[0]:.4f} "
        )
