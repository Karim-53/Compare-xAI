import shap


class Shap:  # todo custom explainers should inhirit from a bigger class
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, f, X):
        self.f = f
        self.X = X
        self.explainer = shap.Explainer(self.f, self.X)

    def explain(self, x):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap:
    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, reference_df, **kwargs):
        self.trained_model = trained_model
        self.f = trained_model.predict
        self.reference_df = reference_df
        print(type(self.reference_df))
        self.explainer = shap.KernelExplainer(self.f, self.reference_df, **kwargs)

    def explain(self, df_to_explain, **kwargs):
        shap_values = self.explainer(df_to_explain)
        self.expected_values = shap_values.base_values
        self.attribution_values = shap_values.values
        self.feature_importance = abs(self.feature_importance).mean(axis=0)


class TreeShap:
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
        self.feature_importance = abs(self.attribution_values).mean(axis=0)
