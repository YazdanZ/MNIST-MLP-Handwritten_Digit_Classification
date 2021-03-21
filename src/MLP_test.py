from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from Loss import softmax_crossentropy_with_logits, grad_softmax_crossentropy_with_logits
from HiddenLayer import HiddenLayer
from Layer import Layer
from Activation import ReLU, Sig, Tanh
from MLP import MLP
import tensorflow as tf
from preprocessing import preprocessing
import visualization
import os

clear = lambda: os.system("clear")


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


preprocessing = preprocessing(
    dataset=tf.keras.datasets.mnist.load_data(),
    symetric_dataset=True,
    dimension_reduction=False,
    data_whitening=False,
)
X_train, y_train, X_test, y_test = preprocessing.preprocess_mnist()

mlp_test_ins = Preprocessing_Test(X_train, y_train, X_test, y_test)
accuracy = mlp_test_ins.test()
clear()
print(f"Sklearn MLP with preprocessed images: {accuracy}\n")


train_log = []
val_log = []

mlp = MLP(ReLU, X_train.shape[1], [100, 200])
gen = mlp.fit(X_train, y_train, 25, 32, X_test, y_test, True)
for epoch, accuracy in enumerate(gen):
    print(f"epoch {epoch}: {accuracy}")
