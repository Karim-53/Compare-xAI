import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from test_superclass import Test
from util import is_ok


def importance_is_b_dummy(importance: list) -> float:
    """ a is the dummy feature """
    if not is_ok(importance):
        return None
    print('importance_is_b_dummy', importance)
    a = abs(importance[0])
    b = abs(importance[1])
    if a == 0. and b > 0.:
        return 1.
    if b == 0. or b < a:
        return 0.
    return round(1. - abs(a/b), 5)


class CorrelatedFeatures(Test):
    # I was not able to make this work see slide 16 / 28
    # https://crossminds.ai/video/problems-with-shapley-value-based-explanations-as-feature-importance-measures-606f4961072e523d7b7811fc/
    name = 'correlated_features'
    ml_task = 'regression'
    description = """Model: output = b^2. Feature a should have a null attribution despite the fact that it is correlated to b. """
    input_features = ['a', 'b']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[i, i] for i in range(6)]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)

        def predict_func(input_features):
            """ input should be a 2d array [data points, args] """
            if isinstance(input_features,pd.DataFrame):
                return input_features['b'] * 5 + input_features['a']
            if isinstance(input_features,np.ndarray):
                return input_features[:, 1] * 5 + input_features[:, 0]
            if isinstance(input_features,list):
                if isinstance(input_features[0], list):
                    return [row[1] * 5 + row[0] for row in input_features]
                else:
                    return input_features[1] * 5 + input_features[0]
            else:
                print('predict_func', input_features, type(input_features))
                return input_features[1] * 5 + input_features[0]

        truth_to_explain = predict_func(dataset_to_explain)
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n  # should I change this? no, because a and b should be perfectly correlated.

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  # todo change it to simple numpy if needed
        self.df_train['target'] = truth_to_explain * n

        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=1, max_depth=3, random_state=0,
                                          base_score=0, eta=1)

        self.X = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)
        self.X_reference = self.X
        self.predict_func = predict_func
        self.predict_proba = None

    @classmethod
    def score(cls, importance=None, **kwargs):
        return {
            'importance_is_b_dummy': importance_is_b_dummy(importance=importance),
            # todo add a test for attribution
        }

def importance_is_c_dummy(importance: list) -> float:
    """ a is the dummy feature """
    if not is_ok(importance):
        return None
    print('importance_is_c_dummy', importance)
    a = abs(importance[0])
    b = abs(importance[1])
    c = abs(importance[2])
    #if c == 0. and b > 0. and a > 0. :
    #    return 1.
    #if b == 0. or b < c:
    #    return 0.
    return round(1. - (min(b, c) / max(b, c)), 5)

class CorrelatedFeaturesNew(Test):
    # I was not able to make this work see slide 16 / 28
    # https://crossminds.ai/video/problems-with-shapley-value-based-explanations-as-feature-importance-measures-606f4961072e523d7b7811fc/
    name = 'correlated_features'
    ml_task = 'regression'
    description = """Model: output = b^2. Feature a should have a null attribution despite the fact that it is correlated to b. """
    input_features = ['a', 'b', 'c']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[np.random.randint(1, 10), i, i] for i in range(6)]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)

        def predict_func(input_features):
            """ input should be a 2d array [data points, args] """
            if isinstance(input_features,pd.DataFrame):
                return input_features['b'] + input_features['c'] + input_features['a']
            if isinstance(input_features,np.ndarray):
                return input_features[:, 1] + input_features[:,2] + input_features[:, 0]
            if isinstance(input_features,list):
                if isinstance(input_features[0], list):
                    return [row[1] + row[2] + row[0] for row in input_features]
                else:
                    return input_features[1] + input_features[2] + input_features[0]
            else:
                print('predict_func', input_features, type(input_features))
                return input_features[1] + input_features[2] + input_features[0]

        truth_to_explain = predict_func(dataset_to_explain)
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n  # should I change this? no, because a and b should be perfectly correlated.

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  # todo change it to simple numpy if needed
        self.df_train['target'] = truth_to_explain * n

        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=1, max_depth=3, random_state=0,
                                          base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)
        self.X_reference = self.X
        self.predict_func = predict_func
        self.predict_proba = None

    @classmethod
    def score(cls, importance=None, **kwargs):
        return {
            'importance_is_c_dummy': importance_is_c_dummy(importance=importance),
            # todo add a test for attribution
        }

class CorrelatedFeaturesInteraction(Test):
    # I was not able to make this work see slide 16 / 28
    # https://crossminds.ai/video/problems-with-shapley-value-based-explanations-as-feature-importance-measures-606f4961072e523d7b7811fc/
    name = 'correlated_features'
    ml_task = 'regression'
    description = """Model: output = b^2. Feature a should have a null attribution despite the fact that it is correlated to b. """
    input_features = ['a', 'b', 'c']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[np.random.randint(1, 10), i, i] for i in range(6)]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)

        def predict_func(input_features):
            """ input should be a 2d array [data points, args] """
            if isinstance(input_features,pd.DataFrame):
                return (input_features['b'] + input_features['c'] + input_features['a']
                        + interaction(input_features['a'], input_features['b'], input_features['c']))
            if isinstance(input_features,np.ndarray):
                return (input_features[:, 1] + input_features[:,2] + input_features[:, 0]
                        + interaction(input_features[:,0], input_features[:,1], input_features[:,2]))
            if isinstance(input_features,list):
                if isinstance(input_features[0], list):
                    return [row[1] + row[2] + row[0] + interaction(row[0], row[1], row[2]) for row in input_features]
                else:
                    return (input_features[1] + input_features[2] + input_features[0] 
                            + interaction(input_features[0], input_features[1], input_features[2]))
            else:
                print('predict_func', input_features, type(input_features))
                return (input_features[1] + input_features[2] + input_features[0]
                        + interaction(input_features[0], input_features[1], input_features[2]))
        
        def interaction(feat1, feat2, feat3):
            first = feat1 * feat2
            second = feat1 * feat3
            return first + second

        truth_to_explain = predict_func(dataset_to_explain)
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n  # should I change this? no, because a and b should be perfectly correlated.

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  # todo change it to simple numpy if needed
        self.df_train['target'] = truth_to_explain * n

        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=1, max_depth=3, random_state=0,
                                          base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)
        self.X_reference = self.X
        self.predict_func = predict_func
        self.predict_proba = None

    @classmethod
    def score(cls, importance=None, **kwargs):
        return {
            'importance_is_c_dummy': importance_is_c_dummy(importance=importance),
            # todo add a test for attribution
        }

if __name__ == '__main__':
    import sys
    sys.path.append('C:/Users/rborn.DESKTOP-LBD00UF/OneDrive/Documents/GitHub/Compare-xAI')
    from src.utils import *
    from explainers.shap_explainer import Shap, TreeShap, ExactShapleyValues
    from explainers.joint_shapley import JointShapley
    from explainers.saabas import Saabas
    import random
    
    np.random.seed(42)
    random.seed(42)

    test = CorrelatedFeaturesNew()
    print(test.truth_to_explain, test.dataset_to_explain)
    #test.df_train['prediction'] = test.trained_model.predict(test.X)
    explainer = JointShapley(test.trained_model, test.trained_model.predict)
    #explainer = ExactShapleyValues(test.predict_func, test.X)
    explainer.explain(test.dataset_to_explain)
    print(test.score(explainer.importance))
    print(test.df_train.drop_duplicates())
    filename = './tmp/' + test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)