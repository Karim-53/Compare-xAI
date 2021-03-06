import pandas as pd

from explainers.explainer_superclass import Explainer


class Sage(Explainer, name='sage'):
    supported_models = ('model_agnostic',)
    # todo SAGE is using the truth to estimate the f importance so we should have this as a selection criteria
    output_importance = True
    source_paper_tag = 'covert2020understanding'
    source_paper_bibliography = r"""@article{covert2020understanding,
  title={Understanding global feature contributions with additive importance measures},
  author={Covert, Ian and Lundberg, Scott M and Lee, Su-In},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={17212--17223},
  year={2020}
}"""
    source_code = 'https://github.com/iancovert/sage'

    description = """Compute feature importance based on Shapley value but faster.
The features that are most critical for the model to make good predictions will have large values ϕ_i(v_f) > 0
While unimportant features will have small values ϕ_i(v_f)≈0
And only features that make the model's performance worse will have negative values ϕ_i(v_f)<0. 
These are SAGE values, and that's SAGE in a nutshell.

Permutation tests, proposed by Leo Breiman for assessing feature importance in random forest models, 
calculate how much the model performance drops when each column of the dataset is permuted [6]. 
SAGE can be viewed as modified version of a permutation test:

Instead of holding out one feature at a time, SAGE holds out larger subsets of features. (By only removing individual 
features, permutation tests may erroneously assign low importance to features with good proxies.)
SAGE draws held out features from their conditional distribution p(X_S¯ ∣ X_S =x_S ) 
rather than their marginal distribution p(X_S¯). (Using the conditional distribution simulates a feature's absence,
whereas using the marginal breaks feature dependencies and produces unlikely feature combinations.)
src: https://iancovert.com/blog/understanding-shap-sage/  

Disadvantage:
The convergence of the algorithm depends on 2 parameters: `thres` and `gap`.
The algorithm can be trapped in a potential infinite loop if we do not fine tune them. """

    def __init__(self, predict_func, X_reference, trained_model=None, **kwargs):
        super().__init__()
        # self.input_names = input_names
        # Set up an imputer to handle missing features
        reference_max_len = min(16, len(X_reference))  # 512 was the default value but I for faster results we use 16
        import sage
        imputer = sage.MarginalImputer(
            predict_func if predict_func is not None else trained_model,
            # it can convert some models into their predict_func :)
            X_reference[:reference_max_len]
        )
        # Set up an estimator
        self.sage_estimator = sage.PermutationEstimator(imputer, 'mse')
        # todo [after submission] get the optimizer from the model or the

    def explain(self, dataset_to_explain, truth_to_explain, **kwargs):
        """

        :param dataset_to_explain:
        :param truth_to_explain: optionnal
        :param kwargs:
        """
        self.expected_values = 'Can not be calculated'
        self.attribution = 'Can not be calculated'
        if truth_to_explain is None:
            self.importance = 'Can not be calculated'
            return
        # if truth_to_explain is not None:
        #     print('explanation_type = SAGE')
        # else:
        #     print('explanation_type = Shapley Effects')
        # todo [after acceptance] re read the sage paper to check the difference

        if isinstance(dataset_to_explain, pd.DataFrame):
            dataset_to_explain = dataset_to_explain.to_numpy()
        if isinstance(truth_to_explain, pd.Series) or isinstance(truth_to_explain, pd.DataFrame):
            truth_to_explain = truth_to_explain.to_numpy()

        # Calculate SAGE values
        print(dataset_to_explain.shape, truth_to_explain.shape)
        print(dataset_to_explain, truth_to_explain)
        sage_values = self.sage_estimator(dataset_to_explain, truth_to_explain,
                                          verbose=True, bar=False,
                                          thresh=.85,
                                          )
        self.importance = sage_values.values
        # sage_values.plot(feature_names)
        # print(sage_values)


if __name__ == "__main__":
    import sage

    print(sage.__version__)

    df = Sage.to_pandas()
    print(df)

    """ Example from https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb """
    import numpy as np
    from sklearn.model_selection import train_test_split

    # Load data
    import sage

    df = sage.datasets.bike()
    feature_names = df.columns.tolist()[:-3]

    # Split data, with total count serving as regression target
    train, test = train_test_split(
        df.values, test_size=int(0.1 * len(df.values)), random_state=123)
    train, val = train_test_split(
        train, test_size=int(0.1 * len(df.values)), random_state=123)
    Y_train = train[:, -1].copy()
    Y_val = val[:, -1].copy()
    Y_test = test[:, -1].copy()
    train = train[:, :-3].copy()
    val = val[:, :-3].copy()
    test = test[:, :-3].copy()

    import xgboost as xgb

    # Set up data
    dtrain = xgb.DMatrix(train, label=Y_train)
    dval = xgb.DMatrix(val, label=Y_val)

    # Parameters
    param = {
        'max_depth': 10,
        'objective': 'reg:squarederror',
        'nthread': 4
    }
    evallist = [(dtrain, 'train'), (dval, 'val')]
    num_round = 50

    # Train
    model = xgb.train(param, dtrain, num_round, evallist, verbose_eval=False)

    # Calculate performance
    mean = np.mean(Y_train)
    base_mse = np.mean((mean - Y_test) ** 2)
    mse = np.mean((model.predict(xgb.DMatrix(test)) - Y_test) ** 2)

    print('Base rate MSE = {:.2f}'.format(base_mse))
    print('Model MSE = {:.2f}'.format(mse))

    # Setup and calculate
    imputer = sage.MarginalImputer(model, test[:16])
    estimator = sage.PermutationEstimator(imputer, 'mse')
    sage_values = estimator(test, Y_test, bar=False)
    importance = sage_values.values
    importance_std = sage_values.std  # also provides std  # todo [after acceptance] include that somewhere
# else:
# from tqdm import tqdm
# from functools import partialmethod
#
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # keeping tqdm is failing
