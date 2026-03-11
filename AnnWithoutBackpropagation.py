import numpy as np

# Define Perceptron Class (No Backpropagation)
class Perceptron:
    def __init__(self, input_size, lr=0.1):
        self.lr = lr
        self.weights = np.zeros(input_size)
        self.bias = 0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        return self.activation(np.dot(X, self.weights) + self.bias)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}")
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                # Update weights and bias
                self.weights += self.lr * error * xi
                self.bias += self.lr * error
                print(f"Input:{xi}, Target:{target}, Prediction:{pred}")
            print("-"*40)

# Training data (AND gate)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,0,0,1])

# Create and train the perceptron
p = Perceptron(input_size=2, lr=0.1)
p.train(X, y, epochs=10)

# Test trained model
print("\nTesting Perceptron")
for xi in X:
    print(f"Input: {xi} → Output: {p.predict(xi)}")
