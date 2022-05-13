import numpy as np

from explainers.explainer_superclass import Explainer, InteractionExplainer
from explainers.interaction_utils import powerset


class StiExplainer(InteractionExplainer):
    def __init__(
            self,
            model,
            input=None,
            baseline=None,
            data_xformer=None,
            output_indices=0,
            batch_size=20,
            verbose=False,
    ):
        InteractionExplainer.__init__(
            self,
            model,
            input,
            baseline,
            data_xformer,
            output_indices,
            batch_size,
            verbose,
        )

    def attribution(self, S, T):
        """
        S: the interaction index set to get attributions for
        T: the input index set
        """
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
            s = len(S)
            att += (-1) ** (w - s) * scores[set_indices[i]]

        return att

    def batch_attribution(
            self, num_orderings, main_effects=False, pairwise=True, seed=None, max_order=2
    ):
        def collect_att(S, S_T_Z_dict, Z_score_dict, n):
            s = len(S)
            subsetsW = powerset(S)

            total_att = 0

            for T in S_T_Z_dict[S]:

                att = 0
                for i, W in enumerate(subsetsW):
                    w = len(W)
                    att += (-1) ** (w - s) * Z_score_dict[S_T_Z_dict[S][T][i]]

                total_att += att

            num_orderings = len(S_T_Z_dict[S])
            return total_att / num_orderings

        num_features = len(self.input)

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
        S_T_Z_dict = dict()
        for S in Ss:
            subsetsW = powerset(S)
            S_T_Z_dict[S] = {}

            if seed is not None:
                np.random.seed(seed)
            for _ in range(num_orderings):
                ordering = np.random.permutation(list(range(num_features)))
                ordering_dict = {ordering[i]: i for i in range(len(ordering))}

                if len(S) == max_order:
                    T = subset_before(S, ordering, ordering_dict)
                else:
                    T = []
                T = tuple(T)

                S_T_Z_dict[S][T] = []

                set_indices = []
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

        #         Z_score_dict = {Z: scores[Z_idx] for Z_idx, Z in enumerate(Z_list)}

        if pairwise:
            res = np.zeros((num_features, num_features))
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    S = (i, j)
                    att = collect_att(S, S_T_Z_dict, Z_score_dict, num_features)
                    res[i, j] = att
            return res

        elif main_effects:
            res = []
            for i in range(num_features):
                S = tuple([i])
                att = collect_att(S, S_T_Z_dict, Z_score_dict, num_features)
                res.append(att)
            return np.array(res)


def subset_before(S, ordering, ordering_dict):
    end_idx = min(ordering_dict[s] for s in S)
    return ordering[:end_idx]


class ShapleyTaylorInteraction(Explainer, name='shapley_taylor_interaction'):
    """ Main wrapper. please use this one """

    supported_models = ('model_agnostic',)
    output_interaction = True

    # is_affected_by_seed = True

    def __init__(self, trained_model, X, nb_features=None, **kwargs):
        super().__init__()
        self.nb_features = nb_features
        if self.nb_features is None:
            self.nb_features = X.shape[1]
        self.sti_method = StiExplainer(trained_model, input=list(X[0]), baseline=list(X[1]), )

    def explain(self, **kwargs):
        self.expected_values = None
        self.attribution = 'Can not be calculated'
        self.importance = 'Can not be calculated'

        inter_atts = self.sti_method.batch_attribution(num_orderings=20, pairwise=True, seed=42)
        inter_scores = []
        for i in range(self.nb_features):
            for j in range(i + 1, self.nb_features):
                S = (i, j)
                inter_scores.append((S, inter_atts[i, j] ** 2))
        self.interaction = inter_scores
