#%%

from random import randint
import matplotlib.pyplot as plt
import seaborn as sb
from ActivationFunctions import *
from Layer import Layer
from NeuralNetwork import NeuralNetwork


def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = np.frombuffer(f.read(16), dtype=">i4")
        images = np.frombuffer(f.read(), dtype=np.uint8)
        return images.reshape(num, rows * cols, 1) / 255.0


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = np.frombuffer(f.read(8), dtype=">i4")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def to_one_hot(labels, dimension=10):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


#%%
# 1 - citanje podataka za treniranje i testiranje
train_images = load_mnist_images('data/digits/train-images-idx3-ubyte/train-images-idx3-ubyte')
train_labels = load_mnist_labels('data/digits/train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images = load_mnist_images('data/digits/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels = load_mnist_labels('data/digits/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# priprema target vektora kao one_hot
train_labels_one_hot = to_one_hot(train_labels)
test_labels_one_hot = to_one_hot(test_labels)

# OVDE BI TREBAO SHAPE da se podesi kako treba

#%%
# prikaz cifre kroz heatmap (zbog pregleda)
entry = randint(0, 5000)
single_digit = train_images[entry]
digit_label = train_labels[entry]
digit_matrix = single_digit.reshape(28,28)
plt.figure(figsize=(8, 6))
sb.heatmap(digit_matrix, annot=False, cmap='binary')
plt.title(f"Heatmap za cifru: {digit_label}")
plt.show()

#%%
# 2 - kreiranje neuronske mreze
learning_rate = 0.01
nn = NeuralNetwork([
    Layer(784, 128, relu, relu_derivative),
    Layer(128, 10, softmax, None)
], learning_rate, method="softmax+cce")
#%%
# 3 - treniranje nad train skupom podataka
nn.train(train_images, train_labels_one_hot)
#%%
# 4 - testiranje
nn.evaluate_model(test_images, test_labels_one_hot)