import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    @abstractmethod
    def forward_propagate(self, input):
        pass

    @abstractmethod
    def backward_propagate(self, input, grad_output):
        pass
