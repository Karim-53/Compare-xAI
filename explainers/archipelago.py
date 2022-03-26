from explainers.archipelago_lib.src.explainer import Archipelago as _Archipelago
from explainers.explainer_superclass import Explainer
from explainers.interaction_utils import proprocess_data
import statsmodels.api as sm
from statsmodels.formula.api import ols


class Archipelago(Explainer):
    """ Main wrapper. please use this one """
    name = 'archipelago'
    interaction = True

    source_code = 'https://github.com/mtsang/archipelago/blob/main/experiments/1.%20archdetect/1.%20synthetic_performance.ipynb'

    # is_affected_by_seed = True

    def __init__(self, trained_model, X, nb_features, **kwargs):
        super().__init__()
        self.nb_features = nb_features
        self.apgo = _Archipelago(trained_model,  input=list(X[0]), baseline=list(X[1]), output_indices=0, batch_size=20)

    def explain(self, **kwargs):
        self.expected_values = None
        self.attribution = 'Can not be calculated'
        self.importance = 'Can not be calculated'

        inter_scores = self.apgo.archdetect()["interactions"]

        self.interaction = inter_scores
