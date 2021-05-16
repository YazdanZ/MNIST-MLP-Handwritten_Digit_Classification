## MNIST Handwritten Digit Classification using a Multi-Layer Perceptron 
![](https://img.shields.io/badge/-python-informational?style=for-the-badge&logo=python&logoColor=white&color=3776AB)

Study conducted as a part of McGill COMP551 (Applied Machine Learning). <br/>
The full paper may be read [here](https://github.com/Unixado/MNIST-MLP-Handwritten_Digit_Classification/blob/master/Report/writeup.pdf).<br/>
Further results can be found in the [Notebook](https://github.com/Unixado/MNIST-MLP-Handwritten_Digit_Classification/blob/master/src/COMP551_Project3.ipynb).


### Abstract
In this project, we studied the performance of **Multi-Layer Perceptrons (MLP) on the MNIST handwritten digits dataset** and ran the MLP model written from scratch. We implemented models with varying depths and widths and studied the train and test accuracies. Although raising depth from zero to two substantially increased the performance of our models, a network depth larger than two offered no advantage. Furthermore, we discovered that our models underfitted
with small network width and plateaued in terms of performance after a width of 60. Next, we studied **non-linearity and
different activation functions**, finding out that ReLU activation enabled the fastest convergence and highest accuracies.
LeakyReLU, Hyperbolic Tangent, and Sigmoid sequentially followed. Next, we showed that adding L2 regularization to the
cost led to similar results compared to the models with no **L2 regularization** applied since our models did not overfit. We
experimented with training models with **normalized and unnormalized images** and observed higher accuracies and faster
convergence rates when the input to the MLP was normalized. Compiling all of our findings, **we configured a model with
a test accuracy of 98% and fast converge after only 10 epoch**.
