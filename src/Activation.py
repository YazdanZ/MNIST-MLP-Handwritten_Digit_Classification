import numpy as np
from Layer import Layer

np.random.seed(0)


class Activation(Layer):
    pass


class LeakyRelu(Activation):
    """Based on this answer on stackoverflow:
    https://stackoverflow.com/questions/50517545/how-do-i-implement-leaky-relu-using-numpy-functions#comment97538129_54206300"""

    def __init__(self, a=0.01):
        self.a = a

    def forward_propagate(self, input):
        return np.where(input > 0, input, input * self.a)

    def backward_propagate(self, input, grad_output):
        gradient = np.where(input > 0, 1.0, self.a)
        return grad_output * gradient


class Tanh(Activation):
    def forward_propagate(self, input):
        return np.tanh(input)

    def backward_propagate(self, input, grad_output):
        gradient = 1.0 - self.forward_propagate(input) * self.forward_propagate(input)
        return grad_output * gradient


class Sig(Activation):
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
