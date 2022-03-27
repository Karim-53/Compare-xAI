from explainers.explainer_superclass import Explainer
from explainers.interaction_utils import proprocess_data
import statsmodels.api as sm
from statsmodels.formula.api import ols


class Anova(Explainer):
    """ Main wrapper. please use this one """
    name = 'anova'
    interaction = True
    supported_models = ('model_agnostic',)
    # is_affected_by_seed = True

    def __init__(self, nb_features, **kwargs):
        super().__init__()
        self.nb_features = nb_features

    def explain(self, dataset_to_explain, truth_to_explain, **kwargs):
        self.expected_values = None
        self.attribution = 'Can not be calculated'
        self.importance = 'Can not be calculated'

        Xs, Ys = proprocess_data(dataset_to_explain, truth_to_explain, valid_size=10000,
                                                              test_size=10000, std_scale_X=True, std_scale=True)
        X_train = Xs["train"]
        Y_train = Ys["train"]

        data = {}
        data['y'] = Y_train.squeeze()
        st = ''
        for i in range(0, X_train.shape[1]):
            data['X' + str(i)] = X_train[:, i]
            st += '+X' + str(i)
        st = "(" + st[1:] + ")"
        formula = 'y ~ ' + st + ":" + st

        lm = ols(formula, data=data).fit()

        table = sm.stats.anova_lm(lm, typ=2)
        inter_scores = []
        for i, name in enumerate(table.index):
            if name == "Residual": continue
            inter = tuple(int(x) for x in name.replace("X", "").split(":"))
            if len(inter) == 1: continue

            inter_scores.append((inter, table.values[i, 0]))

        self.interaction = inter_scores
