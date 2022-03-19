import numpy as np
import shap

from src.utils import get_feature_importance


class Shap:  # todo custom explainers should inhirit from a bigger class
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = shap.Explainer(self.f, self.X)

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap:
    name = 'kernel_shap'
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, X, **kwargs):
        self.trained_model = trained_model
        self.predict_func = trained_model.predict
        self.reference_dataset = np.array(X)
        print(type(self.reference_dataset))
        print(self.reference_dataset.shape)
        self.predict_func(self.reference_dataset)
        self.explainer = shap.KernelExplainer(self.predict_func, self.reference_dataset, **kwargs)

    def explain(self, df_to_explain, **kwargs):
        shap_values = self.explainer.shap_values(np.array(df_to_explain))
        self.expected_values = self.explainer.expected_value
        self.attribution_values = shap_values
        self.feature_importance = get_feature_importance(self.attribution_values)


class TreeShap:
    name = 'tree_shap'
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, df_reference, **kwargs):
        self.trained_model = trained_model
        self.f = trained_model.predict
        self.df_reference = df_reference
        self.explainer = shap.TreeExplainer(self.trained_model, self.df_reference, **kwargs)

    def explain(self, df_to_explain, **kwargs):
        shap_values = self.explainer(df_to_explain, check_additivity=False)
        self.expected_values = shap_values.base_values
        self.attribution_values = shap_values.values
        self.feature_importance = get_feature_importance(self.attribution_values)
