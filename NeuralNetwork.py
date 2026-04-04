import numpy as np

from Layer import Layer

class NeuralNetwork:
    def __init__(self, layers, learning_rate, method="softmax+cce"):
        self.layers = layers
        self.learning_rate = learning_rate
        if method in ["softmax+cce", "mse"]:
            self.method = method
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict(self, x_vector):
        activations = x_vector
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def gradient_step(self, x, y):
        # ovo promeniti. Podaci trebaju biti sredjeni pre treniranja.
        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)

        # 1 - racunanje svih aktivacija na zadati ulaz. Forward pass
        self.predict(x)

        # 2 - racunanje delta pocev od zadnjeg sloja. Backward pass
        self.layers[-1].backward_output(y, self.method)
        for i in reversed(range(len(self.layers)-1)):
            self.layers[i].backward_hidden(self.layers[i+1])

        # 3 - azuriranje tezina. Gradient descent
        for layer in self.layers:
            layer.update(self.learning_rate)

    def evaluate_model(self, test_data, test_labels):
        if np.isnan(test_data).any() or np.isnan(test_labels).any():
            raise ValueError("NaN vrednosti u podacima")
        print("Test started...")
        if self.method == "softmax+cce":
            correct = 0
            for i in range(len(test_data)):
                output = self.predict(test_data[i])
                predicted = np.argmax(output)
                target = np.argmax(test_labels[i])
                if predicted == target:
                    correct += 1
            accuracy = (correct / len(test_data)) * 100
            print(f"Accuracy: {accuracy:6.2f}%")
        elif self.method == "mse":
            mse_sum = 0
            for i in range(len(test_data)):
                predicted = self.predict(test_data[i])
                mse_sum += np.mean((predicted - test_labels[i]) ** 2)
            print(f"MSE: {mse_sum / len(test_data):.6f}")

    def train(self, train_data, train_labels):
        if np.isnan(train_data).any() or np.isnan(train_labels).any():
            raise ValueError("NaN vrednosti u podacima")
        print("Training started...")
        for i in range(len(train_data)):
            if self.method=="softmax+cce":
                self.gradient_step(train_data[i], train_labels[i])
            elif self.method == "mse":
                self.gradient_step(train_data[i], np.array(train_labels[i]))
            if i % 1000 == 0:
                # provera za exploding gradients
                max_w = np.max(np.abs(self.layers[0].weights))
                if np.isnan(max_w):
                    print(f"\nEXPLODED at iteration {i} - weights are NaN")
                    break
                progress = ((i+1)/len(train_data)) * 100
                print(f"\rProgress: {progress:6.2f}% | Max Weight: {max_w:.4f}", end="", flush=True)
        print("\nTraining finished!")