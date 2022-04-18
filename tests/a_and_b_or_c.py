import pandas as pd
from xgboost import XGBRegressor

from tests.test_superclass import Test
from tests.util import importance_xi_more_important


class AAndBOrC(Test):
    name = "a_and_b_or_c"
    description = """Model: A and (B or C)
    Goal make sure that A is more important than B, C
    
    Noise: even if the model output is not == 1. still we expect the xai to give a correct answer => no noise
    """
    input_features = ['A', 'B', 'C']
    dataset_size = 20000

    def __init__(self, **kwargs):
        dataset_to_explain = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                              [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
        self.dataset_to_explain = pd.DataFrame(dataset_to_explain, columns=self.input_features)
        truth_to_explain = [0., 0., 0., 0.,
                            0., 1., 1., 1.]
        self.truth_to_explain = pd.DataFrame(truth_to_explain, columns=['target'])
        n = self.dataset_size // len(self.truth_to_explain)
        X = dataset_to_explain * n

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train  # todo change it to simple numpy if needed
        self.df_train['target'] = truth_to_explain * n

        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=1, max_depth=3, random_state=0,
                                          base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.X_reference = self.X
        self.trained_model.fit(self.X, y=self.df_train.target)
        self.predict_func = self.trained_model.predict

    @staticmethod
    def score(importance=None, **kwargs):
        return {
            'importance_x0_more_important': importance_xi_more_important(importance=importance),
        }


if __name__ == "__main__":
    from src.utils import *

    test = AAndBOrC()
    test.df_train['prediction'] = test.trained_model.predict(test.X)
    print(test.df_train.drop_duplicates())
    filename = './tmp/' + test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)
    # todo add assert if target is different from prediction (move this check to __init__)
