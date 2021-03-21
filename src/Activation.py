import numpy as np
from Layer import Layer


class Activation(Layer):
    pass


class Tanh(Activation):
    def forward_propagate(self, input):
        return np.tanh(input)

    def backward_propagate(self, input, grad_output):
        gradient = 1.0 - self.forward_propagate(input) * self.forward_propagate(input)
        return grad_output * gradient


def Sig(Activation):
    def forward_propagate(self, input):
        return 1 / (1 + np.exp(-input))

    def backward_propagate(self, input, grad_output):
        gradient = self.forward_propagate(input) * (1.0 - self.forward_propagate(input))
        return grad_output * gradient


class ReLU(Activation):
    def forward_propagate(self, input):
        relu_forward_propagate = np.maximum(0, input)
        return relu_forward_propagate

    def backward_propagate(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
