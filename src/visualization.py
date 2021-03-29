import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
    ticks = range(1, epoch_num + 1, 5)
    ax1.plot(range(1, epoch_num + 1), accuracies_test, label="test accuracy")
    ax1.plot(range(1, epoch_num + 1), accuracies_train, label="train accuracy")
    ax1.set_xticks(ticks)
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
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(durations[::5])
    plt.xticks(ticks)
    ax1.grid(True)
    ax1.legend()
    plt.show()


def plot_depth_accuracy(accuracies, depths):
    plt.plot(depths, [train_acc[0] for train_acc in accuracies], label="train accuracy")
    plt.plot(depths, [test_acc[1] for test_acc in accuracies], label="test accuracy")
    plt.xticks(depths)
    plt.xlabel("Depth of the MLP")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_unit_accuracy(accuracies, units):
    plt.plot(units, [train_acc[0] for train_acc in accuracies], label="train accuracy")
    plt.plot(units, [test_acc[1] for test_acc in accuracies], label="test accuracy")
    plt.xticks(units)
    plt.xlabel("Number of units in the hidden layer")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_vs_accuracy(accuracies, lrs, x_label):
    ypos = np.arange(len(lrs))
    plt.xticks(ypos, lrs)
    plt.ylim(0.85, 1.0)
    plt.bar(ypos, [a[0] for a in accuracies], label="train accuracy")
    plt.bar(ypos, [a[1] for a in accuracies], label="test accuracy")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def plot_activation_accuracy(accuracies, activations, title):
    ypos = np.arange(len(activations))
    plt.xticks(ypos, activations)
    plt.bar(ypos, [a[0] for a in accuracies], label="train accuracy")
    plt.bar(ypos, [a[1] for a in accuracies], label="test accuracy")
    plt.xlabel("Activation function")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_confusion_matrix(y_test, pred, title):
    con_matrix = confusion_matrix(y_test, pred)
    df_cm = pd.DataFrame(con_matrix, range(10), range(10))
    sns.heatmap(df_cm, annot=True, cmap="crest")
    plt.ylabel("Predicted digit")
    plt.xlabel("True digit")
    plt.title(title)
    plt.show()
