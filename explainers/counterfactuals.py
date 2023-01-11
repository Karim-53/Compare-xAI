import sys
import traceback
import numpy as np
import shap as shap_lib
import pandas as pd
from sklearn.datasets import make_circles

from explainers.counterfactuals_lib.get_counterfactuals.main_counterfactuals_cxAI import get_counterfactuals_xAI
from explainers.explainer_superclass import Explainer, UnsupportedModelException
from src.utils import get_importance, TimeoutException

def create_data_noise():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df

class DiCE(Explainer, name='DiCE'):
    supported_models = ('model_agnostic',)
    output_counterfactual = True

    def __init__(self, dataset=None, **kwargs):
        # TODO: give it also the model!
        # TODO: make the structure more like shap explainer
        super().__init__()
        self.dataset = dataset

    def explain(self, dataset_to_explain, truth_to_explain, **kwargs): # NOTE: truth_to_explain is input instance
        # print("here, instance:", truth_to_explain)
        # print("get_counterfactuals_xAI(self.dataset, instance)", get_counterfactuals_xAI(self.dataset, truth_to_explain))
        self.counterfactual = get_counterfactuals_xAI(dataset_to_explain, truth_to_explain).to_numpy()
        # print("\nself.cfs\n", self.counterfactual)


if __name__ == '__main__':
    path_circle = r"C:\Users\simon\OneDrive\Sonstiges\Dokumente\GitHub\Compare-xAI\explainers\counterfactuals_lib\unit_tests\data\circles_outlier.csv" # TODO: change to relative path!
    dataset = pd.read_csv(path_circle)
    test = DiCE(dataset)
    out = {'x': [0], 'y': [-1.5]}
    outlier = pd.DataFrame(out, index=[0])
    test.explain(outlier)
    print(test)
