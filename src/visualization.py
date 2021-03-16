import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class visulaization:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def image_show(self):
        plt.imshow(self.x_train[0, :, :, 0])
        plt.colorbar()
        plt.show()

    def digits_distribution(self):
        x = [{i: np.random.randint(1, 5)} for i in range(10)]
        my_colors = [(x / 10.0, x / 20.0, 0.75) for x in range(10)]
        _ = pd.DataFrame(self.y_train).value_counts().plot(kind='bar', stacked=True, color=my_colors)
        plt.show()
