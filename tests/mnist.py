import os
import pickle
from functools import lru_cache

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.utils import root
from tests.test_superclass import Test
from tests.util import importance_dummy, attributions_dummy


@lru_cache(None)
def load_mnist():
    # todo move this to explain because sometimes i just want to instanciate the thing
    # todo [after acceptance] double check that the pointers are not affected when moving to another XAI
    print('Load MNIST...')  # Load data from https://www.openml.org/d/554
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    X = X / 255.0

    # Split data into train partition and test partition
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.7)
    X_reference, X_rest, y_reference, y_rest = train_test_split(X_train, y_train, stratify=y_train, random_state=1,
                                                                test_size=0.999)
    X_to_explain, _, y_to_explain, _ = train_test_split(X_rest, y_rest, stratify=y_rest, random_state=2,
                                                        test_size=0.9995)
    print(len(X_reference), len(X_to_explain))
    mnist_model_path = root + '/tests/tmp/mnist_model.sav'
    if not os.path.exists(mnist_model_path):
        mlp = MLPClassifier(
            hidden_layer_sizes=(40,),
            max_iter=10,
            alpha=1e-4,
            solver="sgd",
            verbose=12,
            random_state=1,
            learning_rate_init=0.2,
        )

        # this example won't converge because of resource usage constraints on
        # our Continuous Integration infrastructure, so we catch the warning and
        # ignore it here
        # from sklearn.exceptions import ConvergenceWarning
        # with warnings.catch_warnings():
        #     warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        mlp.fit(X_train, y_train)

        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Test set score: %f" % mlp.score(X_test, y_test))

        pickle.dump(mlp, open(mnist_model_path, 'wb'))
    else:
        mlp = pickle.load(open(mnist_model_path, 'rb'))
        print("Test set score: %f" % mlp.score(X_test, y_test))

    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    y_train_arr = lb.fit_transform(y_train)
    return X_train, X_reference, X_to_explain, mlp, y_train_arr


class Mnist(Test):
    name = 'mnist'
    ml_task = 'classification'
    description = "code from https://github.com/iancovert/sage-experiments/blob/main/experiments/univariate_predictors.ipynb"
    src = ''
    input_features = [f'px{n}' for n in range(256)]

    def __init__(self):
        self.X, self.X_reference, self.dataset_to_explain, self.trained_model, self.truth_to_explain = load_mnist()

        self.predict_func = self.trained_model.predict
        # self.dataset_size = len(self.X)

    @classmethod
    def score(cls, attribution=None, importance=None, **kwargs):
        if attribution is None and importance is None:
            return {
                'importance_dummy': None,
                'attributions_dummy': None
            }
        # todo assert attribution importance size
        print("score mnist")
        # _dummy_features = self.X.max(axis=0) == 0.
        X, _, _, _, _ = load_mnist()
        _dummy_features = X.max(
            axis=0) == 0.  # todo double check that the MLP is indeed not using these pixels by applying the axiom (brute force)
        return {
            'importance_dummy': importance_dummy(importance=importance, dummy_features=_dummy_features),
            'attributions_dummy': attributions_dummy(attribution=attribution,
                                                     dummy_features=_dummy_features)
        }  # todo [after submission] output the plot showing the False positive etc.
