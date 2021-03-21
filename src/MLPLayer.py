import numpy as np

np.random.seed(42)


class MLPLayer:
    def forward(self, input):
        return input

    def backward(self, input, gradient_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(gradient_output, d_layer_d_input)  # chain rule


class Tanh(MLPLayer):
    def forward(self, input):
        return np.tanh(input)

    def backward(self, input, gradient_output):
        gradient = 1.0 - self.forward(input) * self.forward(input)
        return gradient * gradient_output


class Sig(MLPLayer):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, gradient_output):
        gradient = self.forward(input) * (1.0 - self.forward(input))
        return gradient * gradient_output


class ReLU(MLPLayer):
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, gradient_output):
        gradient = input > 0
        return gradient_output * gradient
