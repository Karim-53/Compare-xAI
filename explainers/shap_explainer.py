import sys
import traceback

import numpy as np
import shap as shap_lib

from explainers.explainer_superclass import Explainer, UnsupportedModelException
from src.utils import get_importance


class Shap(Explainer):
    expected_values = None
    attribution = None
    importance = None

    description = """shap is unifying all other methods (Deeplift, limeshapley regression…)"""

    def __init__(self, f, X, **kwargs):
        self.f = f
        self.X = X
        self.explainer = shap_lib.Explainer(self.f, self.X)

    def explain(self, x, **kwargs):
        shap_values = self.explainer(x)
        self.expected_values, shap_values = shap_values.base_values, shap_values.values
        return shap_values


class KernelShap(Explainer):
    name = 'kernel_shap'
    supported_models = ('model_agnostic',)
    output_attribution = True
    output_importance = True  # inferred

    def __init__(self, trained_model, X, predict_func, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.predict_func = predict_func
        self.reference_dataset = np.array(X, dtype='float64')
        print(self.reference_dataset.dtype)
        # todo find a way to restrict reference dataset size
        # print(type(self.reference_dataset))
        # print(self.reference_dataset.shape)
        self.predict_func(self.reference_dataset)  # just a test
        # todo improve this
        #  WARNING  Using 4629 background data samples could cause slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to summarize the background as K samples.
        try:
            self.explainer = shap_lib.KernelExplainer(self.predict_func, self.reference_dataset, **kwargs)
        except Exception as e:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            print(e)

    def explain(self, dataset_to_explain, **kwargs):
        self.interaction = None
        self.expected_values = self.explainer.expected_value
        shap_values = self.explainer.shap_values(np.array(dataset_to_explain))
        self.attribution = np.array(shap_values)
        self.importance = get_importance(self.attribution)


class TreeShap(Explainer):
    name = 'tree_shap'
    library_version = shap_lib.__version__  # todo [after acceptance] record library name and version automatically

    supported_models = ('tree_based',)
    output_attribution = True
    output_importance = True  # inferred
    description = """ <<In an email exchange, Scott Lundberg clarified that the implicit assumption is that the
features are distributed according to ”the distribution generated by the tree”.>> See https://arxiv.org/pdf/1908.08474.pdf  I don t know if this still holds or was fixed. """

    def __init__(self, trained_model, df_reference, predict_func, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.f = predict_func
        if self.f is None:
            print('why there is no predict_func ?')
            self.f = trained_model.predict
        self.df_reference = df_reference
        try:
            self.explainer = shap_lib.TreeExplainer(self.trained_model, self.df_reference, **kwargs)
        except Exception as e:
            if 'Model type not yet supported by TreeExplainer' in str(e):
                raise UnsupportedModelException(str(e))
            else:
                raise

    def explain(self, dataset_to_explain, **kwargs):
        shap_values = self.explainer(dataset_to_explain,
                                     check_additivity=False)  # todo after acceptance reort all problems to shap due to additivity
        self.expected_values = shap_values.base_values
        self.attribution = shap_values.values
        self.importance = get_importance(self.attribution)


if __name__ == '__main__':
    print(shap_lib.__version__)
