import numpy as np
from tqdm import trange
from Layer import Layer
from Activation import ReLU, Sig, Tanh, Activation
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from HiddenLayer import HiddenLayer


class MLP:
    def __init__(self):
        pass

    def forward(self, network, X):
        activations = []
        input = X

        # Looping through each layer
        for l in network:
            activations.append(l.forward(input))
            # Updating input to last layer output
            input = activations[-1]

        assert len(activations) == len(network)
        return activations

    def fit(self, network, X, y):

        # Get the layer activations
        layer_activations = self.forward(network, X)
        layer_inputs = [
            X
        ] + layer_activations  # layer_input[i] is an input for network[i]
        logits = layer_activations[-1]

        # Compute the loss and the initial gradient
        loss = softmax_crossentropy_with_logits(logits, y)
        loss_grad = grad_softmax_crossentropy_with_logits(logits, y)

        # Propagate gradients through the network
        # Reverse propogation as this is backprop
        for layer_index in range(len(network))[::-1]:
            layer = network[layer_index]

            loss_grad = layer.backward(
                layer_inputs[layer_index], loss_grad
            )  # grad w.r.t. input, also weight updates

        return np.mean(loss)

    def predict(self, network, X):

        logits = self.forward(network, X)[-1]
        return logits.argmax(axis=-1)

    @staticmethod
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.random.permutation(len(inputs))
        for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx : start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]
