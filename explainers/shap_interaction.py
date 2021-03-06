import math
import random

import numpy as np

from explainers.explainer_superclass import Explainer, InteractionExplainer
from explainers.interaction_utils import powerset, random_subset


class SiExplainer(InteractionExplainer):
    """ wrapper from Archipelago repo """

    def __init__(
            self,
            trained_model,
            input=None,
            baseline=None,
            data_xformer=None,
    ):
        output_indices = 0
        batch_size = 20
        verbose = True
        seed = None
        InteractionExplainer.__init__(
            self,
            trained_model,
            input,
            baseline,
            data_xformer,
            output_indices,
            batch_size,
            verbose,
        )
        if seed is not None:
            random.seed(seed)

    def attribution(self, S, num_T):
        """
        S: the interaction index set to get attributions for
        T: the input index set
        """

        s = len(S)
        n = len(self.input)

        N_excl_S = [i for i in range(n) if i not in S]

        num_T = min(num_T, 2 ** len(N_excl_S))

        random_T_set = set()
        for _ in range(num_T):
            T = random_subset(N_excl_S)
            while T in random_T_set:
                T = random_subset(N_excl_S)
            random_T_set.add(T)

        total_att = 0

        for T in random_T_set:
            t = len(T)

            n1 = math.factorial(n - t - s)
            n2 = math.factorial(t)
            d1 = math.factorial(n - s + 1)

            coef = (n1 * n2) / d1

            subsetsW = powerset(S)

            set_indices = []
            for W in subsetsW:
                set_indices.append(tuple(set(W) | set(T)))

            scores_dict = self.batch_set_inference(
                set_indices, self.baseline, self.input, include_context=False
            )
            scores = scores_dict["scores"]

            att = 0
            for i, W in enumerate(subsetsW):
                w = len(W)
                att += (-1) ** (w - s) * scores[set_indices[i]]

            total_att += coef * att

        return total_att

    def batch_attribution(self, num_T, main_effects=False, pairwise=True):
        """
        S: the interaction index set to get attributions for
        T: the input index set
        """

        def collect_att(S, S_T_Z_dict, Z_score_dict, n):
            s = len(S)

            subsetsW = powerset(S)

            total_att = 0

            for T in S_T_Z_dict[S]:

                att = 0
                for i, W in enumerate(subsetsW):
                    w = len(W)
                    att += (-1) ** (w - s) * Z_score_dict[S_T_Z_dict[S][T][i]]

                t = len(T)
                n1 = math.factorial(n - t - s)
                n2 = math.factorial(t)
                d1 = math.factorial(n - s + 1)

                coef = (n1 * n2) / d1
                total_att += coef * att

            return total_att

        n = len(self.input)
        num_features = n

        if main_effects == False and pairwise == False:
            raise ValueError()
        if main_effects == True and pairwise == True:
            raise ValueError()

        Ss = []
        if pairwise:
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    S = (i, j)
                    Ss.append(S)
        elif main_effects:
            for i in range(num_features):
                Ss.append(tuple([i]))

        Z_set = set()
        S_T_Z_dict = {}

        for S in Ss:
            s = len(S)

            N_excl_S = [i for i in range(n) if i not in S]
            num_T = min(num_T, 2 ** len(N_excl_S))

            random_T_set = set()
            for _ in range(num_T):
                T = random_subset(N_excl_S)
                while T in random_T_set:
                    T = random_subset(N_excl_S)
                random_T_set.add(tuple(T))

            S_T_Z_dict[S] = {}

            subsetsW = powerset(S)

            for T in random_T_set:
                S_T_Z_dict[S][T] = []

                for W in subsetsW:
                    Z = tuple(set(W) | set(T))
                    Z_set.add(Z)
                    S_T_Z_dict[S][T].append(Z)

        Z_list = list(Z_set)
        scores_dict = self.batch_set_inference(
            Z_list, self.baseline, self.input, include_context=False
        )
        scores = scores_dict["scores"]
        Z_score_dict = scores

        if pairwise:
            res = np.zeros((num_features, num_features))
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    S = (i, j)
                    att = collect_att(S, S_T_Z_dict, Z_score_dict, n)
                    res[i, j] = att
            return res

        elif main_effects:
            res = []
            for i in range(num_features):
                S = tuple([i])
                att = collect_att(S, S_T_Z_dict, Z_score_dict, n)
                res.append(att)

            return np.array(res)


class ShapInteraction(Explainer, name='shap_interaction'):
    """ Main wrapper. please use this one """

    supported_models = ('model_agnostic',)
    output_interaction = True
    is_affected_by_seed = True

    def __init__(self, trained_model, X, nb_features=None, **kwargs):
        super().__init__()
        self.nb_features = nb_features
        if self.nb_features is None:
            self.nb_features = X.shape[1]
        self.si_method = SiExplainer(trained_model, input=list(X[0]), baseline=list(X[1]), )

    def explain(self, **kwargs):
        self.expected_values = None
        self.attribution = 'Can not be calculated'
        self.importance = 'Can not be calculated'

        num_T = 20
        inter_scores = []
        for i in range(self.nb_features):
            for j in range(i + 1, self.nb_features):
                S = (i, j)
                att = self.si_method.attribution(S, num_T)  # todo [after acceptance] find the tqdm and delete it
                inter_scores.append((S, att ** 2))
        print('inter_scores', inter_scores)
        self.interaction = inter_scores
