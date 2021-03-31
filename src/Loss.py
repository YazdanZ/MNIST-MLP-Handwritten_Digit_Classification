import numpy as np
from scipy.special import softmax

np.random.seed(0)


def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = -logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits - np.max(logits))
    softmax = softmax / np.sum(softmax, axis=-1, keepdims=True)

    return (-ones_for_answers + softmax) / logits.shape[0]


def L2_regularization_loss(weights, biases, lambd, y):
    parameters = {
        "W1": weights,  # (hidden_layer_size[1], input_layer_size),
        "b1": biases  # (hidden_layer_size[1], 1),
    }
    # ,
    # "W2": weights_2,  # (output_layer_size, hidden_layer_size[1]),
    # "b2": biases_2  # (output_layer_size, 1)

    w1 = parameters["W1"]
    m = y.shape[0]

    # Compute L2 regularization cost
    L2_regularization_cost = (np.sum(np.square(w1))) * (lambd / (2 * m))

    # makes sure cost is the dimension we expect.
    L2_regularization_cost = np.squeeze(L2_regularization_cost)
    assert (isinstance(L2_regularization_cost, float))

    return L2_regularization_cost
