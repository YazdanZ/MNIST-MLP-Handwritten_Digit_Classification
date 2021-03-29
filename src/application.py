from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.core.window import Window
from PIL import Image
import numpy as np
from Activation import ReLU, Sig, Tanh, LeakyRelu
from MLP import MLP
import tensorflow as tf
from preprocessing import preprocessing

Window.size = (280, 280)


class Classifier:
    @staticmethod
    def predict_input_img(img):
        print("Fitting model...")
        preprocessing_normalized = preprocessing(
            dataset=tf.keras.datasets.mnist.load_data(),
            symetric_dataset=True,
            dimension_reduction=False,
            data_whitening=False,
            normalize=True,
        )
        X_train, y_train, X_test, y_test = preprocessing_normalized.preprocess_mnist(
            img
        )
        mlp = MLP(ReLU, X_train.shape[1], [128, 128])
        gen = mlp.fit(X_train, y_train, 1, 50)
        ls = list(gen)
        pred = mlp.predict(X_test[0])
        print(f"We believe the input digit is: {pred}")


class Draw(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=10)

    def on_touch_move(self, touch):
        touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        print("Saving canvas...")
        self.export_to_png("img.png")
        img = Image.open("img.png")
        img = img.resize((28, 28))
        img.save("img.png")
        img = np.array(img)
        img = img[:, :, 0]
        Classifier.predict_input_img(img)


class Application(App):
    def build(self):
        return Draw()


if __name__ == "__main__":
    Application().run()
