from neural_network import NeuralNetwork
nn = NeuralNetwork(
    input_size=8,
    hidden_size=8,
    output_size=1,
    learning_rate=0.05
)

nn.summary()

print("\nInput -> Hidden Weights (W1)")
print(nn.W1)

print("\nHidden Bias (b1)")
print(nn.b1)

print("\nHidden -> Output Weights (W2)")
print(nn.W2)

print("\nOutput Bias (b2)")
print(nn.b2)

print("\nTesting ReLU")
print(nn.relu([-5, -2, 0, 3, 8]))

print("\nTesting Sigmoid")
print(nn.sigmoid([-5, -2, 0, 3, 8]))

print("\nTesting ReLU Derivative")
print(nn.relu_derivative([-5, -2, 0, 3, 8]))
