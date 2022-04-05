# import torch
# import torch.nn as nn
# import torch.optim as optim

import numpy as np
import sklearn


# class MLP(nn.Module):
#     def __init__(
#         self,
#         num_features,
#         hidden_units,
#         add_linear=False,
#         act_func=nn.ReLU(),
#     ):
#         super(MLP, self).__init__()
#
#         self.hidden_units = hidden_units
#         self.add_linear = add_linear
#         self.interaction_mlp = create_mlp(
#             [num_features] + hidden_units + [1], act_func=act_func
#         )
#
#         self.add_linear = add_linear
#
#         if add_linear:
#             self.linear = nn.Linear(num_features, 1, bias=False)
#
#     def forward(self, x):
#         y = self.interaction_mlp(x)
#
#         if self.add_linear:
#             y += self.linear(x)
#         return y


def merge_overlapping_sets(
        prediction_scores,
        interaction_atts,
        overlap_thresh=0.5,
        rel_gain_threshold=0,
        patience=1,
        num_features=None,
):
    def overlap_coef(A, B):
        A = set(A)
        B = set(B)
        return len(A & B) / min(len(A), len(B))

    def merge_sets(inter_sets):
        prev_sets = None
        inter_sets = list(inter_sets)
        inter_sets_merged = inter_sets
        while inter_sets != prev_sets:
            prev_sets = list(inter_sets)
            for A in inter_sets:
                for B in inter_sets_merged:
                    if A != B:
                        if overlap_coef(A, B) >= overlap_thresh:
                            inter_sets_merged.append(
                                tuple(sorted(set(A) | set(B)))
                            )  # merge
                            if A in inter_sets_merged:
                                inter_sets_merged.remove(A)
                            if B in inter_sets_merged:
                                inter_sets_merged.remove(B)

            inter_sets = list(set(inter_sets_merged))
        return inter_sets

    def threshold_inter_sets(interaction_atts, prediction_scores):
        scores = prediction_scores
        inter_sets = []
        patience_counter = 0
        best_score = scores[0]
        for i in range(1, len(scores)):
            cur_score = scores[i]
            rel_gain = (cur_score - best_score) / best_score
            inter_sets_temp, _ = zip(*interaction_atts[i - 1])
            if num_features is not None:
                if any(len(inter) == num_features for inter in inter_sets_temp):
                    break
            if rel_gain > rel_gain_threshold:
                best_score = cur_score
                inter_sets = inter_sets_temp
                patience_counter = 0
            else:
                if patience_counter < patience:
                    patience_counter += 1
                else:
                    break
        return inter_sets

    inter_sets = threshold_inter_sets(interaction_atts, prediction_scores)
    inter_sets_merged = merge_sets(inter_sets)

    return inter_sets_merged


######################################################
# The following are based on the official LIME repo
######################################################


def get_sample_distances(Xs):
    all_ones = np.ones((1, Xs["train"].shape[1]))
    Dd = {}
    for k in Xs:
        if k == "scaler":
            continue
        distances = sklearn.metrics.pairwise_distances(
            Xs[k], all_ones, metric="cosine"
        ).ravel()
        Dd[k] = distances

    return Dd


def get_lime_attributions(
        Xs, Ys, max_features=10000, kernel_width=0.25, weight_samples=True, sort=True
):
    from lime import lime_base
    def kernel(d):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

    distances = get_sample_distances(Xs)["train"]
    if not weight_samples:
        distances = np.ones_like(distances).squeeze(1)

    lb = lime_base.LimeBase(kernel_fn=kernel)
    lime_atts = lb.explain_instance_with_data(
        Xs["train"], Ys["train"], distances, 0, max_features
    )[0]
    if sort:
        lime_atts = sorted(lime_atts, key=lambda x: -x[1])
    return lime_atts
