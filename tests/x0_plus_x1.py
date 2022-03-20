""" In this .py we test the effect of the """
import pandas as pd
from xgboost import XGBRegressor

from tests.test_superclass import Test
from tests.util import is_feature_importance_symmetric, is_attribution_values_symmetric


class _X0PlusX1(Test):
    """ This is not a test: It is only a superclass"""
    name = 'x0_plus_x1'
    description = "We test the effect of the train_dataset 's distribution on the explanation.\n"
    input_features = ['x0', 'x1']
    dataset_to_explain = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=input_features)
    truth_to_explain = pd.Series([0, 1, 1, 2], name='target')

    def __init__(self, X):
        self.trained_model = XGBRegressor(objective='reg:squarederror', n_estimators=1, max_depth=2, random_state=0,
                                          base_score=0, eta=1)
        self.dataset_size = len(X)
        self.df_train = pd.DataFrame(X, columns=self.input_features)
        self.df_train['target'] = self.df_train.x0 + self.df_train.x1
        self.X = self.df_train[self.input_features]
        self.df_reference = self.df_train[self.input_features]
        self.trained_model.fit(self.X, y=self.df_train.target)  # == nb of trees
        self.predict_func = self.trained_model.predict

    def score(self, attribution_values=None, feature_importance=None, **kwargs):
        # todo assert attribution_values feature_importance size
        return {
            'is_feature_importance_symmetric': is_feature_importance_symmetric(feature_importance=feature_importance),
            'is_attribution_values_symmetric': is_attribution_values_symmetric(attribution_values=attribution_values)}


class DistributionNonUniformStatDep(_X0PlusX1):
    name = _X0PlusX1.name + "_distribution_non_uniform_stat_dep"
    description = _X0PlusX1.description + "Distribution is non uniform and statistically dependent (both problems)."

    def __init__(self):
        self.X = [[0, 0]] * 50 + [[1, 1]] * 100 + [[1, 0]] * 450 + [[0, 1]] * 1050
        super().__init__(self.X)


def _get_unifrom_stat_dep():
    """ This dataset demostrate
        the problem (SHAP tree_path_dependant gives strange results)
        when (the inputs features are stastically dependant)

        featrues A and B are not statistically independent
        Below is the proof
        From the data we can obtain the folowing probability
        p(A and B)   A=0    A=1
        B=0          0.45  0.05
        B=1          0.05  0.45

                     X=0   X=1
        p(A)         0.5   0.5
        p(B)         0.5   0.5

        Thus p(A and B) != p(A) * p(B)
    """

    X = [[0, 0]] * 4500 + [[1, 1]] * 4500 + [[1, 0]] * 500 + [[0, 1]] * 500
    return X


class DistributionUniformStatDep(_X0PlusX1):
    name = _X0PlusX1.name + "_distribution_uniform_stat_dep"
    description = _X0PlusX1.description + "Distribution is uniform but statistically dependent."

    def __init__(self):
        self.X = _get_unifrom_stat_dep()
        super().__init__(self.X)


def _get_non_uniform_stat_indep(q=0.75):
    """ The problem with the symmetry axiom
    This dataset is supposed to create an issue but besides the difference in the distribution i dont see any other probelm
    https://arxiv.org/pdf/1910.13413.pdf   page 6

    Probability p and q must be different to see the problem

    """

    # q = 0.75  # == p_A_0
    p = 1. - q  # == p_B_0

    assert p < 1
    assert q < 1

    p_0_0 = (1 - p) * (1 - q)
    p_0_1 = (1 - p) * q
    p_1_0 = (1 - q) * p
    p_1_1 = p * q

    assert p_0_0 + p_0_1 + p_1_0 + p_1_1 == 1., (p_0_0, p_0_1, p_1_0, p_1_1)

    p_A_0 = p_0_0 + p_0_1
    p_A_1 = p_1_0 + p_1_1
    p_B_0 = p_0_0 + p_1_0
    p_B_1 = p_0_1 + p_1_1

    assert p_0_0 == p_A_0 * p_B_0
    assert p_0_1 == p_A_0 * p_B_1
    assert p_1_0 == p_A_1 * p_B_0
    assert p_1_1 == p_A_1 * p_B_1
    # ==> the 2 features are statistically independant

    n = 10000  # total number of datapoints

    n_0_0 = int(p_0_0 * n)
    n_0_1 = int(p_0_1 * n)
    n_1_0 = int(p_1_0 * n)
    n_1_1 = int(p_1_1 * n)

    assert n_0_0 + n_0_1 + n_1_0 + n_1_1 == n, "just choose (p,q) and n so that the number of datapoints correspond to the probability"

    X = [[0, 0]] * n_0_0 + [[1, 1]] * n_1_1 + [[1, 0]] * n_1_0 + [[0, 1]] * n_0_1
    return X


class DistributionNonUniformStatIndep(_X0PlusX1):
    name = _X0PlusX1.name + "_distribution_non_uniform_stat_indep"
    description = _X0PlusX1.description + "Distribution is non uniform, statistically independent."

    def __init__(self):
        self.X = _get_non_uniform_stat_indep()
        super().__init__(self.X)


if __name__ == "__main__":
    pass
    # todo double check the tree
    # test = CoughAndFever()
    # test.df_train['prediction'] = test.trained_model.predict(test.X)
    # print(test.df_train.drop_duplicates())
    # filename = test.__class__.__name__ + '.png'
    # plot_tree(test.trained_model, filename)
    # todo add assert if target is different from prediction (move it of __init__

    # todo [after acceptance] asset on the data distribution
    # stat = df_train[input_features].groupby(input_features).size().reset_index().rename(columns={0: 'frequency'})
    # stat['join_distribution'] = stat.frequency.astype(float) / len(df_train)
    #
    # import math
    #
    #
    # def marginal_probability(feature, r):
    #     return sum(df_train[feature] == r[feature]) / len(df_train)
    #
    #
    # stat['proba_product'] = [math.prod([marginal_probability(feature, r) for feature in input_features]) for idx, r in
    #                          stat.iterrows()]
    # stat['equal'] = stat.proba_product == stat.join_distribution
    # print('Is statistically independent:', all(stat['equal']))
    # stat
