import sys
import traceback

import numpy as np
import shap

from explainers.explainer_superclass import Explainer
from src.utils import get_feature_importance


class Shap(Explainer):  # todo custom explainers should inhirit from a bigger class
    expected_values = None
    attribution_values = None
    feature_importance = None

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
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, X, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.predict_func = trained_model.predict
        self.reference_dataset = np.array(X, dtype='float64')
        print(self.reference_dataset.dtype)
        # todo find a way to restrict reference dataset size
        print(type(self.reference_dataset))
        print(self.reference_dataset.shape)
        self.predict_func(self.reference_dataset)  # just a test
        try:
            self.explainer = shap.KernelExplainer(self.predict_func, self.reference_dataset, **kwargs)
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(e)
            self.model_supported = False

    def explain(self, dataset_to_explain, **kwargs):
        if not self.model_supported:
            self.expected_values = None
            self.attribution_values = None
            self.feature_importance = None
            return

        shap_values = self.explainer.shap_values(np.array(dataset_to_explain))
        self.expected_values = self.explainer.expected_value
        self.attribution_values = shap_values
        self.feature_importance = get_feature_importance(self.attribution_values)


class TreeShap(Explainer):
    name = 'tree_shap'
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, X, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.f = trained_model.predict
        self.df_reference = X
        self.model_supported = True
        try:
            self.explainer = shap.TreeExplainer(self.trained_model, self.df_reference, **kwargs)
        except Exception as e:
            if str(e).startswith('Model type not yet supported by TreeExplainer'):
                self.model_supported = False

    def explain(self, dataset_to_explain, **kwargs):
        if not self.model_supported:
            self.expected_values = None
            self.attribution_values = None
            self.feature_importance = None
            return

        shap_values = self.explainer(dataset_to_explain,
                                     check_additivity=False)  # todo after acceptance reort all problems to shap due to additivity
        self.expected_values = shap_values.base_values
        self.attribution_values = shap_values.values
        self.feature_importance = get_feature_importance(self.attribution_values)
