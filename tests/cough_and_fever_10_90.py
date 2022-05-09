import pandas as pd
from xgboost import XGBRegressor

from tests.util import is_ok


class CoughAndFever1090:  # todo [after acceptance] move to the other file and refactor both classes
    name = 'cough_and_fever_10_90'
    ml_task = 'regression'
    input_features = ['Cough', 'Fever']
    dataset_size = 20000
    train_dataset = None
    dataset_to_explain = None
    trained_model = None

    def __init__(self, **kwargs):

        n = self.dataset_size // 4
        X = [[0, 0], [0, 1], [1, 0], [1, 1]] * n

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train
        label = [0., 0., 10., 90.] * n
        self.df_train['target'] = label
        self.dataset_to_explain = self.df_train[self.input_features].iloc[:4]
        self.truth_to_explain = self.df_train.target.iloc[:4]
        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=2, max_depth=2, random_state=0,
                                          base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.X_reference = self.X
        self.trained_model.fit(self.X, y=self.df_train.target)  # == nb of trees
        self.predict_func = self.trained_model.predict

    @staticmethod
    def score(attribution=None, importance=None, **kwargs):
        # todo assert attribution importance size
        def is_cough_more_important_than_fever(importance=None, **kwargs):
            if not is_ok(importance):
                return None

            if importance[0] > importance[1]:
                return 1.
            else:
                return 0.

        def is_cough_attribution_higher_than_fever_attribution(attribution=None, **kwargs):
            if not is_ok(attribution):
                return None
            if attribution[3, 0] > attribution[3, 1]:
                return 1.
            else:
                return 0.

        return {'is_cough_more_important_than_fever': is_cough_more_important_than_fever(
            importance=importance),
            'is_cough_attribution_higher_than_fever_attribution': is_cough_attribution_higher_than_fever_attribution(
                attribution=attribution)}
        # todo add axiom_symmetry


if __name__ == "__main__":
    from src.utils import *

    test = CoughAndFever1090()
    test.df_train['prediction'] = test.trained_model.predict(test.train_dataset)
    print(test.df_train.drop_duplicates())
    filename = test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)
