from neural_network import NeuralNetwork
nn = NeuralNetwork(
    input_size=8,
    hidden_size=8,
    output_size=1,
    learning_rate=0.05
)

nn.summary()

#fake dataset
X = np.array([
    [5, 90, 7, 8, 88, 0, 1, 0],
    [2, 60, 6, 4, 55, 0, 0, 1],
    [8, 95, 8, 10, 92, 0, 1, 0]
])

y = np.array([
    [1],
    [0],
    [1]

# print("\nInput -> Hidden Weights (W1)")
# print(nn.W1)

# print("\nHidden Bias (b1)")
# print(nn.b1)

# print("\nHidden -> Output Weights (W2)")
# print(nn.W2)

# print("\nOutput Bias (b2)")
# print(nn.b2)

# print("\nTesting ReLU")
# print(nn.relu([-5, -2, 0, 3, 8]))

# print("\nTesting Sigmoid")
# print(nn.sigmoid([-5, -2, 0, 3, 8]))

# print("\nTesting ReLU Derivative")
# print(nn.relu_derivative([-5, -2, 0, 3, 8]))


#forwardprop
A2 = nn.forward(X)
print("Predictions")
print(A2)

  #backprop
nn.backward(X, y)
print("\ndZ2")
print(nn.dZ2)

print("\ndW2")
print(nn.dW2)

print("\ndb2")
print(nn.db2)

print("\ndA1")
print(nn.dA1)

print("\ndZ1")
print(nn.dZ1)

print("\ndW1")
print(nn.dW1)

print("\ndb1")
print(nn.db1)





