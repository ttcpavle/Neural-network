import numpy as np


# ako imamo: |sloj1| -W-> |sloj2| matrica W ovde pripada sloju 2
class Layer:
    def __init__(self, input_n, neurons_n, activation_func, activation_derivative):
        # parametri sloja
        self.weights = np.random.randn(neurons_n, input_n) * np.sqrt(2.0 / input_n)
        self.biases = np.zeros((neurons_n, 1))
        self.sigma = activation_func
        self.sigma_derivative = activation_derivative
        self.last_activations = None

        self.z = None
        self.activations = None
        self.deltas = None

    # ulaz: aktivacije poslednjeg sloja
    # izlaz: aktivacija ovog sloja
    def forward(self, last_activations):
        self.last_activations = last_activations
        self.z = self.weights @ self.last_activations + self.biases
        self.activations = self.sigma(self.z)
        return self.activations

    # ulaz: naredni sloj
    # izlaz: menja se deltas za trenutni sloj
    def backward_hidden(self, next_layer):
        self.deltas = (next_layer.weights.T @ next_layer.deltas) * self.sigma_derivative(self.z)

    # ulaz: ocekivana vrednost izlaza i metoda
    # izlaz: azuriranje delta za zadnji sloj, te delte se koriste za slojeve pre njega.
    def backward_output(self, target_vector, method="softmax+cce"):
        if method == "softmax+cce":
            self.deltas = self.activations - target_vector
        elif method == "mse":
            # dC/dz = dC/da * da/dz
            n = self.activations.size
            cost_grad = 2 * (self.activations - target_vector) / n
            self.deltas = cost_grad * self.sigma_derivative(self.z)
        else:
            raise ValueError(f"Unknown method: {method}")

    def update(self, learning_rate):

        grad_w = self.deltas @ self.last_activations.T # delta prethodnog sloja * aktivacije prethodnog sloja
        grad_b = self.deltas # samo delta

        # azuriranje parametara
        self.weights = self.weights - learning_rate * grad_w
        self.biases = self.biases - learning_rate * grad_b