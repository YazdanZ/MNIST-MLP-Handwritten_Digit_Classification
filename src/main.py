import tensorflow as tf
import prerpocessing
import visualization
import MLP_test

prepro_ins = prerpocessing.preprocessing(dataset=tf.keras.datasets.mnist.load_data(), symetric_dataset=True,
                                         dimension_reduction=False, data_whitening=False)
x_train, y_train, x_test, y_test = prepro_ins._prepocess_mnist()

mlp_test_ins = MLP_test.MLP_Test(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
print(mlp_test_ins.test())
