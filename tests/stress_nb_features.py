import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import shap

from tests.test_superclass import Test


class StressNbFeatures(Test):
    name = "stress_nb_features"
    ml_task = 'binary_classification'
    # description = """This is a stress test against a high number of input features (18328) using an nlp model"""

    def __init__(self, **kwargs):
        # Load the IMDB dataset
        corpus, y = shap.datasets.imdb()
        corpus_train, corpus_test, y_train, y_test = train_test_split(corpus, y, test_size=0.0002, random_state=7)

        vectorizer = TfidfVectorizer(min_df=.9,  # minimum frequency
                                     strip_accents='ascii',
                                     lowercase=True,
                                     dtype=np.float32,
                                     )
        X_train = vectorizer.fit_transform(
            corpus_train).toarray()  # sparse also works but Explanation slicing is not yet supported
        X_test = vectorizer.transform(corpus_test).toarray()

        # Fit a linear logistic regression model
        self.trained_model = sklearn.linear_model.LogisticRegression(penalty="l2", C=0.1)
        self.trained_model.fit(X_train, y_train)

        self.input_features = vectorizer.get_feature_names_out()  # get_feature_names
        assert len(self.input_features) > 0, 'no token kept.'
        print(len(self.input_features), 'tokens')
        self.dataset_to_explain = X_test # pd.DataFrame(dataset_to_explain, columns=self.input_features)
        self.truth_to_explain = y_test # pd.DataFrame(truth_to_explain, columns=['target'])

        self.X = X_train  # todo rename X to X_train
        self.X_reference = self.X[:10]
        self.predict_func = self.trained_model.predict

        print(len(self.dataset_to_explain), 'data points to explain')
        print(len(self.X_reference), 'reference points')

    @staticmethod
    def score(importance=None, **kwargs):
        print(importance)
        # return {
        #     'importance_x0_more_important': importance_xi_more_important(importance=importance),
        # }

if __name__ == '__main__':
    # Init test
    test = StressNbFeatures()
    arg = dict(**test.__dict__)
    arg.update(**StressNbFeatures.__dict__)

    from explainers import KernelShap
    explainer = KernelShap(**arg)
    explainer.explain(test.dataset_to_explain)
    print(explainer.importance)
