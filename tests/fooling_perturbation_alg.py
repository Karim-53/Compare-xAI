""" In this .py we test the effect of the distribution on the explanation"""
from tests.test_superclass import Test
try:
    from .fooling_perturbation_alg_lib import *
except ImportError:
    from fooling_perturbation_alg_lib import *


class FoolingPerturbationAlg(Test):
    name = 'fooling_perturbation_alg'
    ml_task = 'binary_classification'
    input_features = ['age', 'two_year_recid', 'priors_count', 'length_of_stay', 'c_charge_degree_F',
                      'c_charge_degree_M', 'sex_Female', 'sex_Male', 'race', 'unrelated_column']


    def __init__(self):
        super().__init__()
        # Get the data set and do some preprocessing
        params = Params()
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
                                          ).train(xtrain, ytrain, feature_names=features,
                                                  categorical_features=categorical_feature_indcs)

        # preparing local explanation
        ex_indc = [0]  # todo add some more datapoints to differenciate f importance and f attribution

        self.trained_model = adv_lime
        self.dataset_size = len(xtrain)
        self.df_train = pd.DataFrame(xtrain, columns=adv_lime.get_column_names())
        self.categorical_feature_indices = categorical_feature_indcs
        self.df_train['target'] = ytrain
        # self.input_features = adv_lime.get_column_names()
        assert self.input_features == adv_lime.get_column_names()
        self.df_reference = self.df_train[self.input_features]
        self.X_reference = xtrain
        self.X = self.X_reference
        self.dataset_to_explain = xtest[ex_indc]
        self.truth_to_explain = ytest[ex_indc]
        self.predict_func = self.trained_model.predict
        self.predict_proba = self.trained_model.predict_proba
        # print(self.input_features)

    @classmethod  # todo change all to class method
    def score(cls, attribution: np.ndarray = None, importance: np.array = None, **kwargs):

        def is_unrelated_feature_important(_importance: np.array):
            """ can receive both global and local importance"""
            if _importance is None or _importance.shape == ():  # case np.asarray(None) see https://stackoverflow.com/questions/54186190/why-does-numpy-ndarray-allow-for-a-none-array
                return None
            print('_importance', _importance, type(_importance))
            # todo MAPLE output _importance = [0, ..., 0] should we consider that it succeeded in the test ? or is there a mistake in the implementation ?
            feature_ranks = np.argsort(abs(_importance))
            print(feature_ranks)
            unrelated_column_index = FoolingPerturbationAlg.input_features.index('unrelated_column')
            unrelated_column_rank = feature_ranks[unrelated_column_index]
            score = 1 - (unrelated_column_rank / (len(cls.input_features) - 1))
            print(score)
            return score
        if attribution is None:
            _attribution = None
        else:
            try:
                _attribution = np.squeeze(attribution)
                if len(_attribution.shape) > 2:
                    print('WARNING fooling_perturbation_alg received _attribution.shape = ', _attribution.shape)
                if len(_attribution.shape) == 2:  # did give an explanation to each class Yes and No class -.-
                    _attribution = _attribution[0, :]
            except:
                _attribution = None
        return {
            'attribution_fragility_is_unrelated_feature_important': is_unrelated_feature_important(_attribution),
            'importance_fragility_is_unrelated_feature_important': is_unrelated_feature_important(importance),
        }


if __name__ == "__main__":
    test = FoolingPerturbationAlg()
    print(test.__dict__.keys())
