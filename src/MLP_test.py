from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import tensorflow as tf
from preprocessing import preprocessing
import visualization


class MLP_Test:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def test(self):
        clf = MLPClassifier(random_state=1, max_iter=3).fit(self.x_train, self.y_train)
        clf.predict_proba(self.x_test)
        clf.predict(self.x_test)
        return clf.score(self.x_test, self.y_test)


prepro_ins = preprocessing(
    dataset=tf.keras.datasets.mnist.load_data(),
    symetric_dataset=True,
    dimension_reduction=False,
    data_whitening=False,
)
x_train, y_train, x_test, y_test = prepro_ins._prepocess_mnist()

mlp_test_ins = MLP_Test(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print(mlp_test_ins.test())
