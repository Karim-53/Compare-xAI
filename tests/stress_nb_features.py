import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap

from tests.test_superclass import Test

# Load the IMDB dataset
corpus, y = shap.datasets.imdb()  # if this fails then
# In `...\Lib\site-packages\shap\datasets.py`
# Replace `with open(cache(github_data_url + "imdb_train.txt")) as f:`
# By `with open(cache(github_data_url + "imdb_train.txt"), encoding="utf8") as f:`
corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.004, random_state=7)

vectorizer = TfidfVectorizer(
    stop_words='english',  # it removes 'not'
    max_df=.999,  # max freq  # increase to keep more tokens
    min_df=.01,  # del tokens if they occur in less than 10% of the docs # decrease to keep more tokens
    strip_accents='ascii',
    lowercase=True,
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(
    corpus_train).toarray()  # sparse also works but Explanation slicing is not yet supported
X_test = vectorizer.transform(corpus_test).toarray()

# Fit a linear logistic regression model
trained_model = sklearn.linear_model.LogisticRegression(
    penalty="l2",
    C=0.1,
    n_jobs=-1,
)
trained_model.fit(X_train, y_train)

input_features = vectorizer.get_feature_names_out()  # get_feature_names
assert len(input_features) > 0, 'no token kept.'
print(len(input_features), 'tokens')

dataset_to_explain = X_test  # pd.DataFrame(dataset_to_explain, columns=self.input_features)
X = X_train
X_reference = X[:100]

print(len(dataset_to_explain), 'data points to explain')
print(len(X_reference), 'reference points')

my_token = 'best'  # also token 'love', 'bad', 'great' should be in the top 10
assert my_token in input_features
my_token_idx = np.where(input_features == my_token)[0][0]


class StressNbFeatures(Test):
    name = "stress_nb_features"
    ml_task = 'binary_classification'
    # description = """This is a stress test against a high number of input features (18328) using an nlp model"""
    input_features = input_features
    trained_model = trained_model

    X = X  # todo rename X to X_train
    X_reference = X_reference

    predict_func = trained_model.predict
    predict_proba = trained_model.predict_proba

    dataset_to_explain = dataset_to_explain
    truth_to_explain = y_test  # pd.DataFrame(truth_to_explain, columns=['target'])

    def __init__(self, **kwargs):
        pass

    @classmethod
    def score(cls, importance=None, **kwargs):
        def importance_token_rank(importance):
            if importance is None:
                return None
            reorder = np.argsort(np.abs(importance))[::-1]  # most import first in np.abs(importance)[ranks]
            print(cls.input_features[reorder][:15])  # top 15 given this explainer
            my_token_rank = np.where(cls.input_features[reorder] == my_token)[0][0]  # lower is more important 0based
            print('my_token_rank', my_token_rank)
            n = len(cls.input_features)
            if my_token_rank < n // 4:
                return 1.
            if my_token_rank > n * 3 / 4:
                return 0.
            return 1. - ( (my_token_rank-n // 4) / (n // 2))

        return {
            'importance_token_rank': importance_token_rank(importance=importance),
        }


# if __name__ == '__main__':
    # Init test
    # test = StressNbFeatures()
    # arg = dict(**test.__dict__)
    # arg.update(**StressNbFeatures.__dict__)

    # from explainers import KernelShap
    # explainer = KernelShap(**arg)
    # explainer.explain(test.dataset_to_explain)
    # print(explainer.importance)
