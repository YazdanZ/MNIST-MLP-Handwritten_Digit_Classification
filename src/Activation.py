import numpy as np
from Layer import Layer


class Activation(Layer):
    pass


class Tanh(Activation):
    def forward(self, input):
        return np.tanh(input)

    def backward(self, input, grad_output):
        gradient = 1.0 - self.forward(input) * self.forward(input)
        return grad_output * gradient


def Sig(Activation):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        gradient = self.forward(input) * (1.0 - self.forward(input))
        return grad_output * gradient


class ReLU(Activation):
    def forward(self, input):
        relu_forward = np.maximum(0, input)
        return relu_forward

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output * relu_grad
