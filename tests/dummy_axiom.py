import numpy as np
import pandas as pd

from tests.test_superclass import Test
from tests.util import is_ok


def importance_is_b_dummy(importance: list) -> float:
    """ a is the dummy feature """
    if not is_ok(importance):
        return None
    a = abs(importance[0])
    b = abs(importance[1])
    if a == 0. and b > 0.:
        return 1.
    if b == 0. or b < a:
        return 0.
    return round(1. - abs(a/b), 5)

class CounterexampleDummyAxiom(Test):
    name = 'counterexample_dummy_axiom'
    ml_task = 'regression'
    description = """Model: output = b^2. Feature a should have a null attribution. This test does not exactly correspond to the one in \cite{sundararajan2020many} because it is using a non uniform distrib that is evaluated in another test."""
    input_features = ['a', 'b']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[i, j] for i in range(6) for j in range(6)]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)

        def predict_func(input_features):
            if isinstance(input_features,pd.DataFrame):
                return input_features['b'] ** 2
            if isinstance(input_features,np.ndarray):
                return input_features[:,1] ** 2
            if isinstance(input_features,list):
                if isinstance(input_features[0], list):
                    return [row[1] ** 2 for row in input_features]
                else:
                    return input_features[1] ** 2
            else:
                print('predict_func', input_features, type(input_features))
                return input_features[1] ** 2

        truth_to_explain = predict_func(dataset_to_explain)
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  # todo change it to simple numpy if needed
        self.df_train['target'] = truth_to_explain * n

        self.trained_model = predict_func
        self.X = self.df_train[self.input_features]
        self.X_reference = self.X
        self.predict_func = predict_func
        self.predict_proba = None

    @classmethod
    def score(cls, importance=None, **kwargs):
        return {
            'importance_is_b_dummy': importance_is_b_dummy(importance=importance),
            # todo add a test for attribution
        }


if __name__ == '__main__':
    test = CounterexampleDummyAxiom()
    print(test.truth_to_explain)