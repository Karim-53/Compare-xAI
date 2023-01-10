import sys
import traceback
import numpy as np
import shap as shap_lib

from explainers.explainer_superclass import Explainer, UnsupportedModelException
from src.utils import get_importance, TimeoutException


class DiCE(Explainer, name='DiCE'):
    supported_models = ('model_agnostic',)
    output_counterfactual = True

    def __init__(self, trained_model, dataset, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        self.dataset = dataset

    def explain(self, **kwargs):
        # counterfactuals = dice.Explainer(self.trained_model, self.dataset)
        # return counterfactuals
        pass
