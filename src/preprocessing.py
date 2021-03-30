import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import pandas as pd


class preprocessing:
    def __init__(
        self, dataset, symetric_dataset, dimension_reduction, data_whitening, normalize
    ):
        """acquire data """
        self.dataset = dataset
        self.x_train_arr = []
        self.x_test_arr = []
        self.symetric = symetric_dataset
        self.dim_reduct = dimension_reduction
        self.data_whitening = data_whitening
        self.normalize = normalize

    def preprocess_mnist(self, img=None, show_digits=False):
        print("Preprocessing the MNIST dataset...\n")

        (x_train, y_train), (x_test, y_test) = self.dataset

        if show_digits:
            width = 5
            height = 5
            rows = 2
            cols = 5
            axes = []
            fig = plt.figure()

            imgs = x_train[:10]
            for a in range(rows * cols):
                axes.append(fig.add_subplot(rows, cols, a + 1))
                plt.imshow(imgs[a])
            fig.tight_layout()
            plt.show()

        if img is not None:
            x_test = img
        self.x_train_arr = self.flatten_maker(x_train)
        self.x_test_arr = self.flatten_maker(x_test)

        if not (self.symetric):
            self.x_train_arr = self.pixel_remover(self.x_train_arr)
            self.x_test_arr = self.pixel_remover(self.x_test_arr)

        """Mean subtraction"""
        self.x_train_arr = self.mean_subtractor(self.x_train_arr)
        self.x_test_arr = self.mean_subtractor(self.x_test_arr)

        if self.normalize:
            """Normalization"""
            self.x_train_arr = self.normalizer(self.x_train_arr)
            self.x_test_arr = self.normalizer(self.x_test_arr)

        """Dimensionality reduction"""
        if self.dim_reduct:
            cov_matrix = (
                np.matmul(self.x_train_arr.T, self.x_train_arr)
                / self.x_train_arr.shape[0]
            )
            U, S, V = np.linalg.svd(cov_matrix)
            Xrot = np.dot(self.x_train_arr, U)
            Xrot_reduced = np.dot(self.x_train_arr, U[:, :100])

        """Data Whitening"""
        if self.data_whitening:
            Xwhite = Xrot / np.sqrt(S + 1e-5)

        return self.x_train_arr, y_train, self.x_test_arr, y_test

    def flatten_maker(self, input_data):
        """flattening data so that
        returns N * D array"""
        output_data = []
        i = 0
        while i < len(input_data):
            output_data.append(np.array(input_data[i]).flatten())
            i += 1
        return np.array(output_data)

    def mean_subtractor(self, input_data):
        """Mean subtraction is done to
        obtain zero-centered data"""
        data = input_data.astype("float32")
        output_data = data - np.mean(data)
        return output_data

    def normalizer(self, nonNorm_data):
        """Normalization through dividing into the
        standard deviation of each dimension (row)"""
        std_dev = np.std(nonNorm_data, axis=1)
        norm_data = nonNorm_data / std_dev[:, None]
        return norm_data

    def pixel_remover(self, data):
        """removing always black (0)
        pixels -where all the numpy
        columns are zero"""
        idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
        output = np.delete(data, idx, axis=1)
        print("Index of removed columns/pixels are " + str(np.array(idx).flatten()))
        return output
