import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.05
    ):

        #make results reproducible
        np.random.seed(42)

        #save hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        #initialize weights
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        z = np.array(z)
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        z = np.array(z)
        return np.maximum(0, z)

    def relu_derivative(self, z):
        z = np.array(z)
        return (z > 0).astype(float)
    #forwardprop
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
    def backward(self, X, y):
        m = X.shape[0]

        self.dZ2 = self.A2 - y 
        self.dW2 = (1 / m) * np.dot(self.A1.T, self.dZ2)
        self.db2 = (1 / m) * np.sum(self.dZ2, axis=0, keepdims=True)

        self.dA1 = np.dot(self.dZ2, self.W2.T)#chain rule

        self.dZ1 = self.dA1 * self.relu_derivative(self.Z1)
        self.dW1 = (1 / m) * np.dot(X.T, self.Z1)
        self.db1 = (1 / m) * np.sum(self.dZ1, axis=0, keepdims=True)

    def update(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1

        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

    def summary(self):

        print("=" * 40)
        print("Neural Network Summary")
        print("=" * 40)

        print(f"Input Neurons  : {self.input_size}")
        print(f"Hidden Neurons : {self.hidden_size}")
        print(f"Output Neurons : {self.output_size}")
        print(f"Learning Rate  : {self.learning_rate}")

        print("\nWeight Shapes")
        print(f"W1 : {self.W1.shape}")
        print(f"b1 : {self.b1.shape}")
        print(f"W2 : {self.W2.shape}")
        print(f"b2 : {self.b2.shape}")

        print("=" * 40)
