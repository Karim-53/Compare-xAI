import numpy as np

from explainers import TreeShap
from tests import CoughAndFever1090

test = CoughAndFever1090()

_explainer = TreeShap(**test.__dict__)
_explainer.explain(test.dataset_to_explain)
# results = test.score(**_explainer.__dict__)

test.dataset_to_explain['shap_pred'] = np.array(_explainer.expected_values) + _explainer.attribution_values.sum(axis=1)
