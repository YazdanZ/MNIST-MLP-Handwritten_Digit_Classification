import numpy as np

np.random.seed(42)


def softmax_cross_entropy(logits, refs):
    logits_ans = logits[np.arange(len(logits)), refs]
    return -logits_ans + np.log(np.sum(np.exp(logits), axis=-1))


def softmax_cross_entropy_grad(logits, refs):
    ones = np.zeros_like(logits)
    ones[np.arange(len(logits)), refs] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    return (-ones + softmax) / logits.shape[0]
