import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from tests.test_superclass import Test
from tests.util import is_ok


def importance_is_c_dummy(importance: list) -> float:
    """ a is the dummy feature """
    if not is_ok(importance):
        return None
    print('importance_is_c_dummy', importance)
    b = abs(importance[1])
    c = abs(importance[2])
    return round(1. - (min(b, c) / max(b, c)), 5)

class CorrelatedFeatures(Test):
    # https://crossminds.ai/video/problems-with-shapley-value-based-explanations-as-feature-importance-measures-606f4961072e523d7b7811fc/
    name = 'correlated_features'
    ml_task = 'regression'
    description = """Features b and c are perfectly correlated. But the model do not use feature c. Explainer should output high importance for only one feature """
    input_features = ['a', 'b', 'c']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[np.random.randint(1, 10), i, i] for i in range(6)]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)

        def predict_func(input_features):
            """ input should be a 2d array [data points, args] """
            if isinstance(input_features,pd.DataFrame):
                return input_features['a'] + input_features['b']
            if isinstance(input_features,np.ndarray):
                return input_features[:, 0] + input_features[:, 1] 
            if isinstance(input_features,list):
                if isinstance(input_features[0], list):
                    return [row[0] + row[1] for row in input_features]
                else:
                    return input_features[0] + input_features[1]
            else:
                print('predict_func', input_features, type(input_features))
                return input_features[0] + input_features[1]

        truth_to_explain = predict_func(dataset_to_explain)
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n  
        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  
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
            #TODO add a test for attribution
        }

if __name__ == '__main__':
    import random
    from xgboost import plot_tree
    import matplotlib.pyplot as plt

    from src.utils import *
    from explainers.shap_explainer import ExactShapleyValues
    
    np.random.seed(42)
    random.seed(42)

    test = CorrelatedFeatures()
    # print(test.truth_to_explain, test.dataset_to_explain)
    # explainer = ExactShapleyValues(test.predict_func, test.X)
    # explainer.explain(test.dataset_to_explain)
    # print(test.score(explainer.importance))
    # print(test.df_train.drop_duplicates())
    #print the image of the trained regression tree

    filename = './tmp/' + test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)
    plt.show()
