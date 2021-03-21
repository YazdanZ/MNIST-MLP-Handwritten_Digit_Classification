import keras
import matplotlib.pyplot as plt
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from HiddenLayer import HiddenLayer
from Layer import Layer
from Activation import ReLU, Sig, Tanh
import numpy as np
from MLP import MLP


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # normalize x
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

## Let's look at some example
plt.figure(figsize=[6, 6])
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.title("Label: %i" % y_train[i])
    plt.imshow(X_train[i].reshape([28, 28]), cmap="gray")
plt.show()

network = []
network.append(HiddenLayer(X_train.shape[1], 100))
network.append(ReLU())
network.append(HiddenLayer(100, 200))
network.append(ReLU())
network.append(HiddenLayer(200, 10))

train_log = []
val_log = []

mlp = MLP()
for epoch in range(25):

    for x_batch, y_batch in mlp.iterate_minibatches(
        X_train, y_train, batchsize=32, shuffle=True
    ):
        mlp.fit(network, x_batch, y_batch)

    train_log.append(np.mean(mlp.predict(network, X_train) == y_train))
    val_log.append(np.mean(mlp.predict(network, X_val) == y_val))

    print("Epoch", epoch)
    print("Train accuracy:", train_log[-1])
    print("Val accuracy:", val_log[-1])
    plt.plot(train_log, label="train accuracy")
    plt.plot(val_log, label="val accuracy")
    plt.legend(loc="best")
    plt.grid()
    plt.show()
