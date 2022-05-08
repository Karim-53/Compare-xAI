""" In this .py we test the effect of the distribution on the explanation"""
from tests.test_superclass import Test
# import warnings
# warnings.filterwarnings('ignore')
# from fooling_lime_shap.adversarial_models import *
import fooling_perturbation_alg_lib as lib


class FoolingPerturbationAlg(Test):
    """ """
    name = 'fooling_perturbation_alg'
    description_short = " "
    # input_features = ['x0', 'x1']
    # dataset_to_explain = pd.DataFrame([[0, 0], [0, 1], [1, 0], [1, 1]], columns=input_features)
    # truth_to_explain = pd.Series([0, 1, 1, 2], name='target')

    description = """
            ### Setup
        Let's begin by examining the COMPAS data set.  This data set consists of defendent information from
        Broward Couty, Florida.Let's suppose that some adversary wants to _mask_ baised or racist behavior on this data set.
    """

    def __init__(self):
        # Get the data set and do some preprocessing
        params = lib.Params("model_configurations/experiment_params.json")
        np.random.seed(params.seed)  # todo this should be deleted
        X, y, cols = get_and_preprocess_compas_data(params)

        # Add a random column -- this is what we'll have LIME/SHAP explain.
        X['unrelated_column'] = np.random.choice([0, 1], size=X.shape[0])
        features = [c for c in X]

        categorical_feature_name = ['two_year_recid', 'c_charge_degree_F', 'c_charge_degree_M', \
                                    'sex_Female', 'sex_Male', 'race', 'unrelated_column']

        categorical_feature_indcs = [features.index(c) for c in categorical_feature_name]

        race_indc = features.index('race')
        unrelated_indcs = features.index('unrelated_column')
        X = X.values

        # Next, let's set up our model f and psi.  f is the what we _actually_ want to classify the data by and psi
        # is what we want LIME/SHAP to explain.
        class racist_model_f:
            # Decision rule: classify negatively if race is black
            def predict(self, X):
                return np.array([params.negative_outcome if x[race_indc] > 0 else params.positive_outcome for x in X])

            def predict_proba(self, X):
                return one_hot_encode(self.predict(X))

            def score(self, X, y):
                return np.sum(self.predict(X) == y) / len(X)

        class innocuous_model_psi:
            # Decision rule: classify according to randomly drawn column 'unrelated column'
            def predict(self, X):
                return np.array(
                    [params.negative_outcome if x[unrelated_indcs] > 0 else params.positive_outcome for x in X])

            def predict_proba(self, X):
                return one_hot_encode(self.predict(X))

            def score(self, X, y):
                return np.sum(self.predict(X) == y) / len(X)

        # Split the data and normalize
        xtrain, xtest, ytrain, ytest = train_test_split(X, y)
        xtest_not_normalized = deepcopy(xtest)
        norm_data = StandardScaler().fit(xtrain).transform
        xtrain = norm_data(xtrain)
        xtest = norm_data(xtest)

        # Train the adversarial model for LIME with f and psi
        adv_lime = Adversarial_Lime_Model(racist_model_f(), innocuous_model_psi()
                                          ).train(xtrain, ytrain, feature_names=features, categorical_features=categorical_feature_indcs)

        # preparing local explanation
        ex_indc = 0  # np.random.choice(xtest.shape[0]) # I fix it to be fair to all xAI

        self.trained_model = adv_lime
        self.dataset_size = len(xtrain)
        self.df_train = pd.DataFrame(xtrain, columns=adv_lime.get_column_names())
        self.categorical_feature_indices = categorical_feature_indcs
        self.df_train['target'] = ytrain
        # self.X = self.df_train[self.input_features]
        self.df_reference = self.df_train[self.input_features]
        self.X_reference = xtrain
        self.dataset_to_explain = xtest[ex_indc]
        self.truth_to_explain = ytest[ex_indc]
        self.predict_func = self.trained_model.predict_proba
        self.input_features = adv_lime.get_column_names()

    @staticmethod
    def score(attribution=None, importance=None, **kwargs):
        # todo assert attribution importance size
        return {
            # 'importance_symmetric': importance_symmetric(importance=importance),
            # 'is_attribution_symmetric': is_attribution_symmetric(attribution=attribution)
        }


if __name__ == "__main__":
    FoolingPerturbationAlg()



    pass
    # todo double check the tree
    # test = CoughAndFever()
    # test.df_train['prediction'] = test.trained_model.predict(test.X)
    # print(test.df_train.drop_duplicates())
    # filename = test.__class__.__name__ + '.png'
    # plot_tree(test.trained_model, filename)
    # todo add assert if target is different from prediction (move it to __init__

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
