import pandas as pd
from xgboost import XGBRegressor

from tests.util import is_feature_importance_symmetric, is_attribution_values_symmetric


class CoughAndFever:  # (Metric):
    name = "cough_and_fever"
    # todo add last_update = version of the release
    # todo [after submission] refactor
    input_features = ['Cough', 'Fever']
    dataset_size = 20000
    dataset_to_explain = None
    trained_model = None
    predict_func = None

    def __init__(self, **kwargs):

        n = self.dataset_size // 4
        X = [[0, 0], [0, 1], [1, 0], [1, 1]] * n

        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_reference = self.df_train #  todo change it to simple numpy if needed
        label = [0., 0., 0., 80.] * n
        self.df_train['target'] = label
        self.dataset_to_explain = self.df_train[self.input_features].iloc[:4]
        self.truth_to_explain = self.df_train.target.iloc[:4]
        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=2, max_depth=2, random_state=0 ,base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)  # == nb of trees
        self.predict_func = self.trained_model.predict

    def score(self, attribution_values=None, feature_importance=None, **kwargs):
        # todo assert attribution_values feature_importance size
        return {
            'is_feature_importance_symmetric': is_feature_importance_symmetric(feature_importance=feature_importance),
            'is_attribution_values_symmetric': is_attribution_values_symmetric(attribution_values=attribution_values)}
        # todo add axiom_symmetry


if __name__ == "__main__":
    from src.utils import *
    test = CoughAndFever()
    test.df_train['prediction'] = test.trained_model.predict(test.X)
    print(test.df_train.drop_duplicates())
    filename = test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)
    # todo add assert if target is different from prediction (move it of __init__
