import numpy as np
from tqdm import trange
from Loss import softmax_cross_entropy, softmax_cross_entropy_grad
from HiddenLayer import HiddenLayer
from MLPLayer import MLPLayer, Sig, ReLU, Tanh

np.random.seed(42)


class MLP:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        activations = []
        input = X

        for layer in self.layers:
            activations.append(layer.forward(input))
            input = activations[-1]

        return activations

    def predict(self, X):
        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def fit(self, X, y):
        activations = self.forward(X)
        inputs = [X] + activations
        logits = activations[-1]

        # find initial gradient and loss
        loss = softmax_cross_entropy(logits, y)
        loss_gradient = softmax_cross_entropy_grad(logits, y)

        # backpropagation through the network
        for index in range(len(self.layers))[::-1]:
            layer = self.layers[index]
            loss_gradient = layer.backward(inputs[index], loss_gradient)

        return np.mean(loss)

    def minibatch_iterator(self, X, y, batch_size):
        indices = np.random.permutation(len(X))

        for index in trange(0, len(X) - batch_size + 1, batch_size):
            ex = indices[index : index + batch_size]
            yield X[ex], y[ex]
