from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from Activation import ReLU, Sig, Tanh
from MLP import MLP
import tensorflow as tf
from preprocessing import preprocessing
from visualization import plot_epoch_accuracy
import matplotlib.pyplot as plt


class Preprocessing_Test:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def test(self):
        """Test preprocessing using the Sklearn MLPClassifier"""
        clf = MLPClassifier(random_state=1, max_iter=3).fit(self.x_train, self.y_train)
        clf.predict_proba(self.x_test)
        clf.predict(self.x_test)
        return clf.score(self.x_test, self.y_test)


preprocessing_normalized = preprocessing(
    dataset=tf.keras.datasets.mnist.load_data(),
    symetric_dataset=True,
    dimension_reduction=False,
    data_whitening=False,
    normalize=True,
)
X_train, y_train, X_test, y_test = preprocessing_normalized.preprocess_mnist()

mlp_test_ins = Preprocessing_Test(X_train, y_train, X_test, y_test)
accuracy = mlp_test_ins.test()
print(f"Sklearn MLP with preprocessed images: {accuracy}\n")


def test_mlp_model(mlp, epochs, batchsize, print_lists=False):
    gen = mlp.fit(X_train, y_train, epochs, batchsize, X_test, y_test, True)
    ls = list(gen)
    accuracies_test = [l[0] for l in ls]
    accuracies_train = [l[1] for l in ls]
    durations = [round(l[2]) for l in ls]
    if print_lists:
        for i in range(epochs):
            epoch = i
            accuracy_train = accuracies_train[i]
            accuracy_test = accuracies_test[i]
            duration = durations[i]
            print(
                f"epoch {epoch} - train: {accuracy_train}, test: {accuracy_test}, duration: {duration}"
            )

    plot_epoch_accuracy(accuracies_test, accuracies_train, durations, epochs)
    train_acc = mlp.evaluate_acc(mlp.predict(X_train), y_train)
    test_acc = mlp.evaluate_acc(mlp.predict(X_test), y_test)

    print(f"\nTrain accuracy after {epochs} epochs: {train_acc}")
    print(f"Test accuracy after {epochs} epochs: {test_acc}")


# Part3.1
# ---------------------------------------------------------------------
#  print("\n\nmlp with no hidden layer, directly map input to output")
#  mlp = MLP(ReLU, X_train.shape[1], [])
#  test_mlp_model(mlp, 40, 30, print_lists=True)
#
#  print("\n\nmlp a sigle hidden layer having 128 units with ReLU activation")
#  mlp = MLP(ReLU, X_train.shape[1], [128])
#  test_mlp_model(mlp, 40, 30, print_lists=True)

#  print("\n\nmlp a 2 hidden layers having 128 units with ReLU activation")
#  mlp = MLP(ReLU, X_train.shape[1], [128, 128])
#  test_mlp_model(mlp, 40, 30, print_lists=True)
# ---------------------------------------------------------------------
# Part3.2
#  print("\n\nmlp a 2 hidden layers having 128 units with TanH activation")
#  mlp = MLP(Tanh, X_train.shape[1], [128, 128])
#  test_mlp_model(mlp, 40, 30, print_lists=True)
#
#  print("\n\nmlp a 2 hidden layers having 128 units with Sigmoid activation")
#  mlp = MLP(Sig, X_train.shape[1], [128, 128])
#  test_mlp_model(mlp, 40, 30, print_lists=True)


preproc = preprocessing(
    dataset=tf.keras.datasets.mnist.load_data(),
    symetric_dataset=True,
    dimension_reduction=False,
    data_whitening=False,
    normalize=False,
)
X_train, y_train, X_test, y_test = preproc.preprocess_mnist()


print(
    "\n\nmlp a 2 hidden layers having 128 units with ReLU activation (unnormalized images)"
)
mlp = MLP(ReLU, X_train.shape[1], [128, 128])
test_mlp_model(mlp, 40, 30, print_lists=True)
