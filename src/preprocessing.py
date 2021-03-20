import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
import pandas as pd


class preprocessing:
    x_train_arr = []

    def __init__(self, dataset):
        """acquire data """
        self.dataset = dataset

    def _prepocess_mnist(self):
        i = 1
        print("Preprocessing the MNIST dataset...\n")

        (x_train, y_train), (x_test, y_test) = self.dataset

        while i < len(x_train):
            self.x_train_arr.append(np.array(x_train[i]).flatten())
            i += 1
        self.x_train_arr = np.array(self.x_train_arr)
        """removing always black (0) 
        pixels -where all the numpy 
        columns are zero"""
        idx = np.argwhere(np.all(self.x_train_arr[..., :] == 0, axis=0))
        self.x_train_arr = np.delete(self.x_train_arr, idx, axis=1)

        print("Index of removed columns/pixels are " + str(np.array(idx).flatten()))

        """Mean subtraction is done to 
        obtain zero-centered data"""
        self.x_train_arr = self.x_train_arr.astype('float32')
        self.x_train_arr -= np.mean(self.x_train_arr)

        """Normalization through dividing into the
         standard deviation of each dimension (row)"""
        std_dev = np.std(self.x_train_arr, axis=1)
        self.x_train_arr = self.x_train_arr / std_dev[:, None]

        """Dimensionality reduction through PCA, 
        data covariance matrix and eigen vectors"""
        cov_matrix = np.matmul(self.x_train_arr.T, self.x_train_arr) / self.x_train_arr.shape[0]
        U, S, V = np.linalg.svd(cov_matrix)
        Xrot = np.dot(self.x_train_arr, U)
        Xrot_reduced = np.dot(self.x_train_arr, U[:, :100])

        """Whitening the data"""
        Xwhite = Xrot / np.sqrt(S + 1e-5)

        return Xrot_reduced, Xwhite
