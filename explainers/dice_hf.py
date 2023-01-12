import sys
import traceback
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

from explainers.counterfactuals_lib.apply import apply_all_metrics
from explainers.counterfactuals_lib.get_counterfactuals.main_counterfactuals_cxAI import get_dice_cxAI
from explainers.explainer_superclass import Explainer, UnsupportedModelException
from src.utils import get_importance, TimeoutException


def create_data_noise():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df


class DiCE_HF(Explainer, name='DiCE_HF'):
    supported_models = ('model_agnostic',)
    output_counterfactual = True

    def __init__(self, dataset=None, **kwargs):
        # NOTE: Do I need to give it the model here too? (model is created in explain(...))
        # TODO: make the structure more like shap explainer?
        super().__init__()
        self.dataset = dataset

    def explain(self, dataset_to_explain, truth_to_explain, **kwargs):  # NOTE: truth_to_explain is input instance
        print("here, instance:", truth_to_explain)
        # print("get_counterfactuals_xAI(self.dataset, instance)", get_dice_cxAI(dataset_to_explain, truth_to_explain))
        cfs_dice = get_dice_cxAI(dataset_to_explain, truth_to_explain)
        truth_to_explain['label'] = 0
        _, cfs_optimized, _ = apply_all_metrics(dataset_to_explain, truth_to_explain, cfs_dice)
        self.counterfactual = cfs_optimized.to_numpy()
        print("\nself.cfs\n", self.counterfactual)


if __name__ == '__main__':
    dataset = create_data_noise()
    test = DiCE_HF(dataset)
    out = {'x': [0], 'y': [-1.5]}
    outlier = pd.DataFrame(out, index=[0])
    test.explain(dataset, outlier)
    print(test)
