from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from Activation import ReLU, Sig, Tanh, LeakyRelu
from MLP import MLP
import tensorflow as tf
from preprocessing import preprocessing
from visualization import (
    plot_epoch_accuracy,
    plot_depth_accuracy,
    plot_unit_accuracy,
    plot_activation_accuracy,
    plot_lr_accuracy,
    plot_confusion_matrix,
)
import matplotlib.pyplot as plt

np.random.seed(0)


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


def test_mlp_model(
    mlp, epochs, batchsize, print_lists=False, plot=True, get_predictions=False
):
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
                f"epoch {epoch + 1} - train: {accuracy_train}, test: {accuracy_test}, duration: {duration}"
            )
    if plot:
        plot_epoch_accuracy(accuracies_test, accuracies_train, durations, epochs)
    train_pred = mlp.predict(X_train)
    test_pred = mlp.predict(X_test)
    train_acc = mlp.evaluate_acc(train_pred, y_train)
    test_acc = mlp.evaluate_acc(test_pred, y_test)
    duration = durations[-1]
    if print_lists:
        print(f"\nTrain accuracy after {epochs} epochs: {train_acc}")
        print(f"Test accuracy after {epochs} epochs: {test_acc}")
        print(f"Duration of {epochs} epochs: {duration}")
    if get_predictions:
        return test_pred, train_pred
    return train_acc, test_acc


# Part3.1
# ---------------------------------------------------------------------
#  print("\n\nmlp with no hidden layer, directly map input to output")
#  print("minibatch sizes: 30, learning_rate=0.1")
#  mlp = MLP(ReLU, X_train.shape[1], [])
#  test_mlp_model(mlp, 51, 30, print_lists=True)  # Figure1
#
#  print("\n\nmlp a sigle hidden layer having 128 units with ReLU activation")
#  print("minibatch sizes: 30, learning_rate=0.1")
#  mlp = MLP(ReLU, X_train.shape[1], [128])
#  test_mlp_model(mlp, 51, 30, print_lists=True)  # Figure2
#
#  print("\n\nmlp a 2 hidden layers having 128 units with ReLU activation")
#  print("minibatch sizes: 30, learning_rate=0.1")
#  mlp = MLP(ReLU, X_train.shape[1], [128, 128])
#  test_mlp_model(mlp, 51, 30, print_lists=True)  # Figure3
#  ---------------------------------------------------------------------


# ---------------------------------------------------------------------
#  Part3.2
#  accuracies = []
#
#  print("\n\nmlp a 2 hidden layers having 128 units with ReLU activation")
#  mlp = MLP(ReLU, X_train.shape[1], [128, 128])
#  accuracies.append(test_mlp_model(mlp, 40, 30, print_lists=True))  # Figure6
#
#  print("\n\nmlp a 2 hidden layers having 128 units with LeakyRelu activation")
#  mlp = MLP(LeakyRelu, X_train.shape[1], [128, 128])
#  accuracies.append(test_mlp_model(mlp, 40, 30, print_lists=True))  # Figure19
#
#  print("\n\nmlp a 2 hidden layers having 128 units with TanH activation")
#  mlp = MLP(Tanh, X_train.shape[1], [128, 128])
#  accuracies.append(test_mlp_model(mlp, 40, 30, print_lists=True))  # Figure7
#
#  print("\n\nmlp a 2 hidden layers having 128 units with Sigmoid activation")
#  mlp = MLP(Sig, X_train.shape[1], [128, 128])
#  accuracies.append(test_mlp_model(mlp, 40, 30, print_lists=True))  # Figure8
#
#
#  activations = ["ReLu", "LeakyRelu", "TanH", "Sigmoid"]
#  title = "Accuracy of the 2 layer MLP using different activation functions"
#  plot_activation_accuracy(accuracies, activations, title)  # Figure9
# ---------------------------------------------------------------------
# Part3.3

# ---------------------------------------------------------------------
# Part3.4

#  preproc = preprocessing(
#  dataset=tf.keras.datasets.mnist.load_data(),
#  symetric_dataset=True,
#  dimension_reduction=False,
#  data_whitening=False,
#  normalize=False,
#  )
#  X_train, y_train, X_test, y_test = preproc.preprocess_mnist()
#
#
#  print(
#  "\n\nmlp a 2 hidden layers having 128 units with ReLU activation (unnormalized images)"
#  )
#  mlp = MLP(ReLU, X_train.shape[1], [128, 128])
#  test_mlp_model(mlp, 5, 30, print_lists=True)

# ---------------------------------------------------------------------
#  Extra experiements
#  def test_depth_accuracy(depth, epochs):
#  accuracies = []
#  depths = range(depth)
#  for depth in depths:
#  mlp = MLP(ReLU, X_train.shape[1], [128] * depth)
#  accuracies.append(test_mlp_model(mlp, epochs, 30, plot=False))
#  plot_depth_accuracy(accuracies, depths)


#  def test_units_accuracy(units, steps, epochs):
#  accuracies = []
#  units = range(1, units, steps)
#  for unit in units:
#  mlp = MLP(ReLU, X_train.shape[1], [unit])
#  accuracies.append(test_mlp_model(mlp, epochs, 30, plot=False))
#  plot_unit_accuracy(accuracies, units)


def tune_learning_rate():
    accuracies = []
    lrs = [0.01, 0.1, 1, 10]  # Figure10, 11, 12, 13
    for lr in lrs:
        print("\n\nmlp a 2 hidden layers having 128 units with ReLU activation")
        print(f"minibatch sizes: 30, learning_rate={lr}")
        mlp = MLP(ReLU, X_train.shape[1], [128, 128], lr=lr)
        accuracies.append(test_mlp_model(mlp, 40, 30, print_lists=True))
    plot_lr_accuracy(accuracies, lrs)


def tune_batchsize():
    accuracies = []
    bsizes = [10, 100, 1000]  # Fig15, 16, 17
    for bsize in bsizes:
        print("\n\nmlp a 2 hidden layers having 128 units with ReLU activation")
        print(f"minibatch sizes: {bsize}, learning_rate={0.1}")
        mlp = MLP(ReLU, X_train.shape[1], [128, 128])
        accuracies.append(test_mlp_model(mlp, 30, bsize, print_lists=True))
    plot_lr_accuracy(accuracies, bsizes)


def tune_hyperparams():
    bsizes = [30, 50, 70, 100]
    lrs = [0.01, 0.1]
    accs = []
    pairs = []
    for bsize in bsizes:
        for lr in lrs:
            pairs.append((lr, bsize))
            mlp = MLP(ReLU, X_train.shape[1], [128, 128], lr=lr)
            _, test_acc = test_mlp_model(mlp, 1, bsize, print_lists=False, plot=False)
            accs.append(test_acc)
            print(f"test accuracy: {test_acc}, learning rate: {lr}, batchsize: {bsize}")

    max_index = accs.index(max(accs))
    lr, bs = pairs[max_index]
    print(f"Best hyperparameter: learning rate = {lr}, {bs}")


tune_hyperparams()

#  tune_learning_rate()  # Figure14

#  tune_batchsize()  # Figure18

#  test_depth_accuracy(17, 25)  # Figure4
#  test_units_accuracy(220, 20, 25) # Figure5


def confusion_matrix(mlp, epochs):
    test_prediction, train_prediction = test_mlp_model(
        mlp, epochs, 30, print_lists=False, plot=False, get_predictions=True
    )
    plot_confusion_matrix(
        y_test, test_prediction, "Test data confusion_matrix"
    )  # Figure20
    plot_confusion_matrix(
        y_train, train_prediction, "Train data confusion_matrix"
    )  # Figure21
