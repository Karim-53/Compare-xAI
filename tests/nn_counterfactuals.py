import pandas as pd
from xgboost import XGBRegressor
from sklearn.datasets import make_circles

from tests.counterfactuals_lib.proxy_cxAI import get_f1_score
from tests.test_superclass import Test
from sklearn.svm import SVC
from tests.util import importance_symmetric, is_attribution_symmetric


def create_data():
    X, y = make_circles(n_samples=1000, random_state=1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df


def create_data_noise():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df


def create_data_imbalanced(t=0):
    X_1, y_1 = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=1)
    df_1 = pd.DataFrame(dict(x=X_1[:, 0], y=X_1[:, 1], label=y_1))
    X_2, y_2 = make_circles(n_samples=1000, noise=0.05, factor=0.5)
    df_2 = pd.DataFrame(dict(x=X_2[:, 0], y=X_2[:, 1], label=y_2))
    df_2 = df_2.drop(df_2[df_2.label == t].index)
    df_3 = pd.concat([df_1, df_2], ignore_index=True)
    return df_3


class Circle(Test):
    name = "1nn_circle"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data()
    trained_model = None
    predict_func = None  # TODO: do I need this?

    def __init__(self, truth_to_explain=pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleNoise(Test):
    name = "1nn_circle_noise"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_noise()
    trained_model = None
    predict_func = None  # TODO: do I need this?

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleOutlier(Test):
    name = "1nn_circle_outlier"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_noise()
    trained_model = None
    predict_func = None  # TODO: do I need this?

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1.5]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1.5], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleBigTarget(Test):
    name = "1nn_circle_big_target"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_imbalanced()
    trained_model = None
    predict_func = None  # TODO: do I need this?

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_imbalanced()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleSmallTarget(Test):
    name = "1nn_circle_small_target"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_imbalanced(1)
    trained_model = None
    predict_func = None  # TODO: do I need this?

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_imbalanced(1)
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


if __name__ == '__main__':
    df = create_data_imbalanced(1)
    print("counts\n", df['label'].value_counts())
    # test = CircleNoise()
    # print(test)
    # out = {'x': [0], 'y': [-1.5]}
    # outlier = pd.DataFrame(out, index=[0])
    # print(test.predict_func(outlier))
