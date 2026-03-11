import numpy as np

# Activation and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Network parameters
np.random.seed(1)
input_neurons, hidden_neurons, output_neurons = 2, 2, 1
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))
lr = 0.1
epochs = 10000

# Training using Backpropagation
for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_input)

    # Compute error
    error = y - predicted_output

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    d_hidden = d_output.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_output)

    # Update weights and biases
    weights_hidden_output += hidden_output.T.dot(d_output) * lr
    weights_input_hidden += X.T.dot(d_hidden) * lr
    bias_output += np.sum(d_output, axis=0, keepdims=True) * lr
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * lr

# Test trained network
print("Final Predictions After Training:\n")
for xi in X:
    hidden = sigmoid(np.dot(xi, weights_input_hidden) + bias_hidden)
    output = sigmoid(np.dot(hidden, weights_hidden_output) + bias_output)
    print(f"Input: {xi} → Predicted Output: {output.round(3)}")
