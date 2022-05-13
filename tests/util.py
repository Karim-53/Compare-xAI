from typing import Iterable

import numpy as np


def is_ok(values, dim=None):
    if values is None or isinstance(values, str):
        return False
    if not isinstance(values, Iterable):
        return False
    if dim is None:
        return True
    if isinstance(values, list) and len(dim) == 1 and len(values) == dim[0]:
        return True
    else:
        return values.shape == dim


def importance_symmetric(importance):
    if not is_ok(importance):
        return None
    diff = abs(importance[0] - importance[1])
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.


def importance_xi_more_important(importance, i=0):
    if not is_ok(importance):
        return None
    score = 1.
    n = len(importance)
    for j in range(n):
        if i == j:
            continue
        if importance[i] < importance[j]:
            score -= 1. / (n - 1)

    return score


def is_attribution_symmetric(attribution):
    if not is_ok(attribution):
        return None
    # print(attribution)
    diff = np.max(np.abs(attribution[:, 0] - attribution[[0, 2, 1, 3], 1]))
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.


def importance_dummy(importance, dummy_features: list):
    if not is_ok(importance):
        return None
    return sum([fi == 0. for fi in importance[dummy_features]]) / len(dummy_features)


def attributions_dummy(attribution, dummy_features):
    if not is_ok(attribution):
        return None
    max_attributions = attribution.__abs__().max(axis=0)
    return sum([a == 0. for a in max_attributions[dummy_features]]) / len(dummy_features)
