import tensorflow as tf
import prerpocessing
import visualization

prepro_ins = prerpocessing.preprocessing(tf.keras.datasets.mnist.load_data())
Xrot_reduced, Xwhite = prepro_ins._prepocess_mnist()

