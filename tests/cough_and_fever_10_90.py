import pandas as pd
from xgboost import XGBRegressor


class CoughAndFever1090:  # (Metric):
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
        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=2, max_depth=2, random_state=0 ,base_score=0, eta=1)
        self.X = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)  # == nb of trees
        self.predict_func = self.trained_model.predict

    def score(self, attribution_values=None, feature_importance=None, **kwargs):
        # todo assert attribution_values feature_importance size
        def is_cough_more_important_than_fever(feature_importance=None, **kwargs):
            if feature_importance is None:
                return None

            if feature_importance[0] > feature_importance[1]:
                return 1.
            else:
                return 0.

        def is_cough_attribution_higher_than_fever_attribution(attribution_values=None, **kwargs):
            if attribution_values[3, 0] > attribution_values[3, 1]:
                return 1.
            else:
                return 0.

        return {'is_cough_more_important_than_fever': is_cough_more_important_than_fever(**locals()),
                'is_cough_attribution_higher_than_fever_attribution': is_cough_attribution_higher_than_fever_attribution(
                    **locals())}
        # todo add axiom_symmetry


if __name__ == "__main__":
    from src.utils import *

    test = CoughAndFever1090()
    test.df_train['prediction'] = test.trained_model.predict(test.train_dataset)
    print(test.df_train.drop_duplicates())
    filename = test.__class__.__name__ + '.png'
    plot_tree(test.trained_model, filename)
