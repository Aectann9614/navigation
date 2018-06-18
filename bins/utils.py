import numpy as np


def iter2vector(iterable):
    vector = np.zeros((len(iterable), 1), float)
    for i in enumerate(iterable):
        vector[i] = iterable[i]
    return vector


def vector2tuple(vector):
    def generator():
        for i in range(vector.shape[0]):
            yield vector[i, 0]
    return tuple(generator())
