from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.core.window import Window
from PIL import Image
import numpy as np

Window.size = (100, 100)


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
        img = img[:, :, 0].flatten()
        print(img)


class Application(App):
    def build(self):
        return Draw()


if __name__ == "__main__":
    Application().run()
