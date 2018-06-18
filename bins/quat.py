import numpy as np
from math import sqrt


def is_quat(that):
    if isinstance(that, np.ndarray):
        if that.shape == (4,) and that.dtype == float:
            return True
    return False


def quat(quat_params):
    if len(quat_params) == 4 and isinstance(quat_params[0], float) and isinstance(quat_params[1], float) \
            and isinstance(quat_params[2], float) and isinstance(quat_params[3], float):
        return np.array([quat_params[0], quat_params[1], quat_params[2], quat_params[3]], float)
    raise TypeError("Params of quaternion must be float")


def zero_quat():
    return quat((0.0, 0.0, 0.0, 0.0))


def extend_matrix(that):
    """
    Returns extended quaternion representation, how matrix form, for multiply
    :param that: np.ndarray - quaternion
    :return: np.ndarray - matrix
    """
    return np.array([[[that[0], -that[1], -that[2], -that[3]]],
                     [that[1], that[0], that[3], -that[2]],
                     [that[2], -that[3], that[0], that[1]],
                     [that[3], that[2], -that[1], that[0]]], float)


def transform_matrix(that):
    """
    Returns transform matrix by quaternion
    :param that: np.ndarray - quaternion
    :return: np.ndarray - matrix
    """
    return np.array([[that[0] ** 2 + that[1] ** 2 - that[2] ** 2 - that[3] ** 2,
                      2 * (that[1] * that[2] - that[0] * that[3]),
                      2 * (that[1] * that[3] + that[0] * that[2])],
                     [2 * (that[1] * that[2] + that[0] * that[3]),
                      that[0] ** 2 + that[2] ** 2 - that[1] ** 2 - that[3] ** 2,
                      2 * (that[2] * that[3] + that[0] * that[1])],
                     [2 * (that[2] * that[3] - that[0] * that[1]),
                      2 * (that[2] * that[3] + that[0] * that[1]),
                      that[0] ** 2 + that[3] ** 2 - that[1] ** 2 - that[2] ** 2]], float)


def is_normal(that, eps=0.001):
    return 1 - (that ** 2).sum() < eps


def normalize(that, eps=0.001):
    if is_normal(that, eps):
        return that
    else:
        return that / sqrt((that ** 2).sum())


def multiply(that, other):
    return np.dot(extend_matrix(that), other.copy().resize(4, 1))
