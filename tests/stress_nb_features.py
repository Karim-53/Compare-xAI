import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap

from tests.test_superclass import Test

# Load the IMDB dataset
corpus, y = shap.datasets.imdb()
corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.0002, random_state=7)

vectorizer = TfidfVectorizer(
    max_df=.9,  # max freq
    min_df=.9,  # minimum frequency
    strip_accents='ascii',
    lowercase=True,
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(
    corpus_train).toarray()  # sparse also works but Explanation slicing is not yet supported
X_test = vectorizer.transform(corpus_test).toarray()


class StressNbFeatures(Test):
    name = "stress_nb_features"
    ml_task = 'binary_classification'
    # description = """This is a stress test against a high number of input features (18328) using an nlp model"""

    # Fit a linear logistic regression model
    trained_model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
    trained_model.fit(X_train, y_train)

    input_features = vectorizer.get_feature_names_out()  # get_feature_names
    assert len(input_features) > 0, 'no token kept.'
    print(len(input_features), 'tokens')

    dataset_to_explain = X_test  # pd.DataFrame(dataset_to_explain, columns=self.input_features)
    truth_to_explain = y_test  # pd.DataFrame(truth_to_explain, columns=['target'])

    X = X_train  # todo rename X to X_train
    X_reference = X[:10]
    predict_func = trained_model.predict

    print(len(dataset_to_explain), 'data points to explain')
    print(len(X_reference), 'reference points')

    assert len(np.where(input_features == 'bad')[0])
    token_bad_idx = np.where(input_features == 'bad')[0][0]

    def __init__(self, **kwargs):
        pass

    @classmethod
    def score(cls, importance=None, **kwargs):
        def importance_token_rank(importance):
            token_bad_rank = importance[cls.token_bad_idx]
            if token_bad_rank < 10:
                return 1.
            if token_bad_rank > 100:
                return 0.
            return 1. - (token_bad_rank / 100)

        return {
            'importance_token_rank': importance_token_rank(importance=importance),
        }


if __name__ == '__main__':
    # Init test
    test = StressNbFeatures()
    arg = dict(**test.__dict__)
    arg.update(**StressNbFeatures.__dict__)

    from explainers import KernelShap
    explainer = KernelShap(**arg)
    explainer.explain(test.dataset_to_explain)
    print(explainer.importance)
