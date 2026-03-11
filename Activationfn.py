import numpy as np
def binary_step(x):
    return np.where(x >= 0, 1, 0)
def linear(x):
    return x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.maximum(0, x)
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
def elu(x, alpha=1):
    return np.where(x > 0, x, alpha*(np.exp(x)-1))
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()
x = np.array([-2, -1, 0, 1, 2])
print("Input:", x)
print("Binary Step:", binary_step(x))
print("Linear:", linear(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))
print("ReLU:", relu(x))
print("Leaky ReLU:", leaky_relu(x))
print("ELU:", elu(x))
print("Softmax:", softmax(x))
