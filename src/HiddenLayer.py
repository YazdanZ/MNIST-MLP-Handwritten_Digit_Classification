from MLPLayer import MLPLayer
import numpy as np

np.random.seed(42)


class HiddenLayer(MLPLayer):
    def __init__(self, input, output, learning_rate=0.1):
        self.lr = learning_rate
        self.bias = np.zeros(output)
        self.weight = np.random.normal(
            loc=0.0, scale=np.sqrt(2 / (input + output)), size=(input, output)
        )

        def forward(self, input):
            return np.dot(input, self.weight) + self.bias

        def backward(self, input, gradient_output):
            grad_in = np.dot(gradient_output, self.weight.T)
            grad_weight = np.dot(input.T, gradient_output)
            grad_biases = gradient_output.mean(axis=0) * input.shape[0]

            self.weight -= self.lr * grad_weight
            self.bias -= self.lr * grad_biases

            return grad_in
