import tensorflow as tf
import prerpocessing
import visualization

prepro_ins = prerpocessing.preprocessing(dataset=tf.keras.datasets.mnist.load_data(), symetric_dataset=True,
                                         dimension_reduction=False, data_whitening=False)
x_train, y_train, x_test, y_test = prepro_ins._prepocess_mnist()
