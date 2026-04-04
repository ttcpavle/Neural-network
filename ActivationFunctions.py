from abc import abstractmethod

import numpy as np


def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    exp_z = np.exp(z - np.max(z)) # Stabilan softmax
    return exp_z / exp_z.sum(axis=0, keepdims=True)

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)