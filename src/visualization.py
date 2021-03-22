import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class visualization:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def image_show(self):
        plt.imshow(self.x_train[0, :, :, 0])
        plt.colorbar()
        plt.show()

    def digits_distribution(self):
        x = [{i: np.random.randint(1, 5)} for i in range(10)]
        my_colors = [(x / 10.0, x / 20.0, 0.75) for x in range(10)]
        _ = (
            pd.DataFrame(self.y_train)
            .value_counts()
            .plot(kind="bar", stacked=True, color=my_colors)
        )
        plt.show()


def plot_epoch_accuracy(accuracies_test, accuracies_train, durations, epoch_num):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    fig.subplots_adjust(bottom=0.2)
    ax1.plot(range(epoch_num), accuracies_test, label="test accuracy")
    ax1.plot(range(epoch_num), accuracies_train, label="train accuracy")
    ax1.set_xticks(range(epoch_num))
    ax1.set_xlabel("epoch")
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    ax2.spines["bottom"].set_position(("axes", -0.2))
    ax2.set_frame_on(True)
    ax2.patch.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.spines["bottom"].set_visible(True)
    ax2.set_xlabel("duration")
    ax2.set_xticks(range(epoch_num))
    ax2.set_xticklabels(durations)
    plt.xticks(np.arange(0, epoch_num, 1.0))

    ax1.legend()
    plt.grid()
    plt.show()
