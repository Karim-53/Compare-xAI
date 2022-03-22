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


def is_feature_importance_symmetric(feature_importance):  # feature_importance=None,
    if not is_ok(feature_importance):
        return None
    diff = abs(feature_importance[0] - feature_importance[1])
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.


def is_attribution_values_symmetric(attribution_values):
    if not is_ok(attribution_values):
        return None
    print(attribution_values)
    diff = np.max(np.abs(attribution_values[:, 0] - attribution_values[[0, 2, 1, 3], 1]))
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.
def attributions_dummy(attribution_values, dummy_features):
    if not is_ok(attribution_values):
        return None
    max_attributions = attribution_values.__abs__().max(axis=0)
    return sum([a == 0. for a in max_attributions[dummy_features]]) / len(dummy_features)