import numpy as np


def is_feature_importance_symmetric(feature_importance=None, **kwargs):
    if feature_importance is None:
        return None
    diff = abs(feature_importance[0] - feature_importance[1])
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.


def is_attribution_values_symmetric(attribution_values, **kwargs):
    diff = np.max(np.abs(attribution_values[:, 0] - attribution_values[[0, 2, 1, 3], 1]))
    if diff < 1:
        return 1. - diff  # if there is a small epsilon then it's gonna hit the score
    else:
        return 0.
