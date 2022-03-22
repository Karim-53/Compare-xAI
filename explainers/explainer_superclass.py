import numpy as np
from sklearn import neural_network

MODELS = ['tree_based', 'neural_network']
EXTENDED_MODELS = {'model_agnostic': MODELS}


def supported_models_developed(supported_models):
    _supported_models_developed = list(supported_models)
    for e in supported_models:
        _supported_models_developed += EXTENDED_MODELS.get(e, [])
    return _supported_models_developed

class Explainer:
    name = None
    description = None
    supported_models = ()



    # todo [after acceptance] add complexity as str
    # todo [after acceptance] add last_update = version of the release of this repo
    # todo add source paper just the bibtex tag
    # todo add a pretty way to print the class

    expected_values = None  # keep them saved here to know what could be calculated
    attribution_values = None
    feature_importance = None

    def explain(self, x, **kwargs):  # todo [after acceptance] change this to __call__ ?
        from src.explainer import valid_explainers
        raise NotImplementedError(
            f"This explainer is not supported at the moment. Explainers supported are {[e.name for e in valid_explainers]}"
        )


class Random(Explainer):
    name = 'baseline_random'
    description = 'This is not a real explainer it helps measure the baseline score and processing time.'
    supported_models = ('model_agnostic',)
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, **kwargs):
        super().__init__()

    def explain(self, dataset_to_explain, **kwargs):
        # todo [after acceptance] with np seed = 0
        arr = np.array(dataset_to_explain)
        self.expected_values = np.random.randn(arr.shape[0])
        self.attribution_values = np.random.randn(*arr.shape)
        self.feature_importance = np.random.randn(arr.shape[1])
