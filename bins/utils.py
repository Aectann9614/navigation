import numpy as np
from itertools import tee


def iter2vector(iterable):
    vector = np.zeros((len(iterable), 1), float)
    for i in enumerate(iterable):
        vector[i] = iterable[i]
    return vector


def vector2tuple(vector):
    if len(vector.shape) == 1:
        return tuple([vector[i] for i in range(vector.shape[0])])
    elif len(vector.shape) == 2 and vector.shape[1] == 1:
        return tuple([vector[i, 0] for i in range(vector.shape[0])])
    raise ValueError("Param vector is not vector")


def window(lst, criteria, map_func=None):
    window_list = []
    for elem in lst:
        window_list.append(elem)
        if criteria(window_list):
            if map_func is not None:
                yield (map_func(e) for e in window_list)
            else:
                yield window_list
            window_list = []


def window_count(lst, count, map_func=None):
    return window(lst, lambda l: len(l) >= count, map_func)


def split_tee(lst, col_num):
    def split(l, count):
        return (e[count] for e in l)
    tee_lists = tee(lst, col_num)
    return tuple((split(tee_lists[i], i) for i in range(col_num)))
