import numpy as np


class Explainer:
    name = None
    description = None
    # todo [after acceptance] add complexity as str
    # todo [after acceptance] add last_update = version of the release of this repo
    # todo add source paper just the bibtex tag
    # todo add a pretty way to print the class

    expected_values = None
    attribution_values = None
    feature_importance = None

    def explain(self, x):
        raise NotImplementedError(
            f"This explainer is not supported at the moment. Explainers supported are {[e.name for e in valid_explainers]}"
        )


class Random(Explainer):
    name = 'baseline_random'
    description = 'This is not a real explainer it helps measure the baseline score and processing time.'

    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, **kwargs):
        pass  # todo [after acceptance] do I really need that

    def explain(self, x):
        # todo [after acceptance] with np seed = 0
        arr = np.array(x)
        self.expected_values = np.random.randn(arr.shape[0])
        self.attribution_values = np.random.randn(*arr.shape)
        self.feature_importance = np.random.randn(arr.shape[1])
