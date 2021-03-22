import numpy as np
from time import time
from tqdm import trange
from Layer import Layer
from Activation import ReLU, Sig, Tanh, Activation
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from HiddenLayer import HiddenLayer


class MLP:
    def __init__(self, activation, num_in, hidden, num_out=10):
        network = []

        # no hidden layers
        if len(hidden) == 0:
            network.append(HiddenLayer(num_in, num_out))

        for index, layer in enumerate(hidden):

            # input layer
            if index == 0:
                network.append(HiddenLayer(num_in, layer))
                network.append(activation())

            # deep layer
            if index != len(hidden) - 1:
                network.append(HiddenLayer(layer, hidden[index + 1]))
                network.append(activation())
            else:
                # output layer
                network.append(HiddenLayer(layer, num_out))

        self.network = network

    def forward_propagate(self, X):
        activations = []
        input = X

        # Looping through each layer
        for l in self.network:
            activations.append(l.forward_propagate(input))
            # Updating input to last layer output
            input = activations[-1]

        assert len(activations) == len(self.network)
        return activations

    def fit(
        self, X, y, epoch_num, batchsize, X_test=None, y_test=None, show_progress=False
    ):
        start_time = time()
        for epoch in range(epoch_num):
            for x_batch, y_batch in self.iterate_minibatches(
                X, y, batchsize=batchsize, shuffle=True
            ):
                self._fit(x_batch, y_batch)
            if show_progress:
                ex_time = time() - start_time
                yield self.evaluate_acc(
                    y_test, self.predict(X_test)
                ), self.evaluate_acc(y, self.predict(X)), ex_time

    def _fit(self, X, y):

        # Get the layer activations
        layer_activations = self.forward_propagate(X)
        layer_inputs = [
            X
        ] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, y)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

        # Propagate gradients through the network
        # Reverse propogation as this is backprop
        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]

            loss_grad = layer.backward_propagate(
                layer_inputs[layer_index], loss_grad
            )  # grad w.r.t. input, also weight updates

        return np.mean(loss)

    def predict(self, X):

        logits = self.forward_propagate(X)[-1]
        return logits.argmax(axis=-1)

    @staticmethod
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.random.permutation(len(inputs))
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @staticmethod
    def evaluate_acc(true_labels, target_labels):
        """Evaluate the model accuracy""" ""
        return np.mean(true_labels == target_labels)
