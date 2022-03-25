import sys
import traceback

import numpy as np
import shap

from explainers.explainer_superclass import Explainer
from src.utils import get_importance


class Shap(Explainer):  # todo custom explainers should inhirit from a bigger class
    expected_values = None
    attribution = None
    importance = None

    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = shap.Explainer(self.f, self.X)

    def explain(self, x, **kwargs):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap(Explainer):
    name = 'kernel_shap'
    supported_models = ('model_agnostic',)
    attribution = True
    importance = True  # inferred

    def __init__(self, trained_model, X, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.predict_func = trained_model.predict
        self.reference_dataset = np.array(X, dtype='float64')
        print(self.reference_dataset.dtype)
        # todo find a way to restrict reference dataset size
        # print(type(self.reference_dataset))
        # print(self.reference_dataset.shape)
        self.predict_func(self.reference_dataset)  # just a test
        self.model_supported = True
        try:
            self.explainer = shap.KernelExplainer(self.predict_func, self.reference_dataset, **kwargs)
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(e)
            self.model_supported = False

    def explain(self, dataset_to_explain, **kwargs):
        self.interaction = None
        if not self.model_supported:
            self.expected_values = None
            self.attribution = None
            self.importance = None
            return

        self.expected_values = self.explainer.expected_value
        shap_values = self.explainer.shap_values(np.array(dataset_to_explain))
        self.attribution = shap_values
        self.importance = get_importance(self.attribution)


class TreeShap(Explainer):
    name = 'tree_shap'
    supported_models = ('tree_based',)
    attribution = True
    importance = True  # inferred

    def __init__(self, trained_model, X, predict_func=None, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.f = predict_func
        if self.f is None:
            self.f = trained_model.predict
        self.df_reference = X
        self.model_supported = True
        try:
            self.explainer = shap.TreeExplainer(self.trained_model, self.df_reference, **kwargs)
        except Exception as e:
            if str(e).startswith('Model type not yet supported by TreeExplainer') or str(e).startswith(
                    'Unsupported masker type'):
                self.model_supported = False

    def explain(self, dataset_to_explain, **kwargs):
        if not self.model_supported:
            self.expected_values = None
            self.attribution = None
            self.importance = None
            return

        shap_values = self.explainer(dataset_to_explain,
                                     check_additivity=False)  # todo after acceptance reort all problems to shap due to additivity
        self.expected_values = shap_values.base_values
        self.attribution = shap_values.values
        self.importance = get_importance(self.attribution)
