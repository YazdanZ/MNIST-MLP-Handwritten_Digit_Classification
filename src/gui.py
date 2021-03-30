from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.core.window import Window
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from Activation import ReLU, Sig, Tanh, LeakyRelu
from MLP import MLP
import tensorflow as tf
from preprocessing import preprocessing
import pyfiglet
import os

Window.size = (280, 280)


class Classifier:
    def __init__(self):
        self.mlp = None

    def predict_input_img(self, img):
        plt.imshow(img, cmap="hot", interpolation="nearest")
        plt.show()
        preprocessing_normalized = preprocessing(
            dataset=tf.keras.datasets.mnist.load_data(),
            symetric_dataset=True,
            dimension_reduction=False,
            data_whitening=False,
            normalize=True,
        )
        X_train, y_train, X_test, y_test = preprocessing_normalized.preprocess_mnist(
            [img]
        )
        if self.mlp is None:
            print("Training the model.\nThis will only happen once.")
            self.mlp = MLP(ReLU, X_train.shape[1], [128, 128])
            gen = self.mlp.fit(X_train, y_train, 25, 50)
            ls = list(gen)
        pred = self.mlp.predict(X_test)
        print(f"Our MultiLayer Perceptron thinks the digit is: {pred[0]}")
        ascii_banner = pyfiglet.figlet_format(str(pred[0]))
        print(ascii_banner)


class Draw(Widget):
    classifier = Classifier()

    def on_touch_down(self, touch):
        with self.canvas:
            os.system("cls" if os.name == "nt" else "clear")
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=1)

    def on_touch_move(self, touch):
        touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        print("Saving canvas...")
        self.export_to_png("../demo/img.png")
        img = Image.open("../demo/img.png")
        img = img.resize((28, 28))
        img.save("../demo/img.png")
        img = np.array(img)
        img = img[:, :, 0]
        Draw.classifier.predict_input_img(img)
        self.canvas.clear()


class Application(App):
    def build(self):
        return Draw()


if __name__ == "__main__":
    Application().run()
