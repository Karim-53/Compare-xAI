import torch
import shap
import numpy as np
import pandas as pd 
import random
import sys
from IPython.display import display, clear_output
import copy

from .explainer_superclass import Explainer


class JointShapley(Explainer, name='joint_shapley'):
    supported_models = ('model_agnostic')
    output_attribution = True
    output_importance = True
    output_interaction = True

    def __init__(self, trained_model, predict_func = None, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        if predict_func is None:
           self.value_f = self.trained_model.predict
        else:
            predict_func = predict_func

        def predict_func_numpy(X):
            if isinstance(X, pd.DataFrame):
                return predict_func(X.values)
            else:
                return predict_func(X)

        self.value_f = predict_func_numpy


        

    def explain(self, dataset_to_explain, k = 2, num_iter = 500, verbose=True, **kwargs):
        if isinstance(dataset_to_explain, pd.DataFrame) is False:
            dataset_to_explain = pd.DataFrame(dataset_to_explain)
        n_features = frozenset(list(dataset_to_explain.columns))
        #num_obs, n = dataset_to_explain.shape
        coalitions_to_k = list(get_powerset_to_k(n_features, k))
        if verbose:
            print(f"Starting calculation for k = {k} with {len(coalitions_to_k)} features")

        local_joint_shapleys = pd.DataFrame(index = dataset_to_explain.index, columns=coalitions_to_k)

        for cln_n, cln in enumerate(coalitions_to_k):
            if verbose:
               display(f"cln = {cln_n + 1} / {len(coalitions_to_k)}")
            value_function = (
                lambda cln: get_estimate_for_coalition(cln, num_iter, k, dataset_to_explain, self.value_f)
            )
            local_joint_shapleys.loc[:, [cln]] = value_function(cln).reshape(-1,1)
        self.attribution = calc_attribution(local_joint_shapleys, n_features)
        self.importance = np.abs(self.attribution).mean(axis=0)
        self.interaction = local_joint_shapleys


### Code from https://github.com/harris-chris/joint-shapley-values/blob/main/boston-housing-walkthrough.ipynb
def get_powerset_to_k(
    features, 
    k,
    init=True,
):
    if init:
        features = [s if type(s) == frozenset else frozenset([s]) for s in features]
    if len(features) <= 1:
        yield features[0]
        yield frozenset()
    else:
        for item in get_powerset_to_k(features[1:], k, False):
            if len(item) <= k - 1:
                yield features[0].union(item)
            yield item
            
def get_powerset_to_k_ex_emptyset(seq, k):
    gen = get_powerset_to_k(seq, k)
    for item in gen:
        if item != frozenset():
            yield item


def get_coalition_arrivals(t_cln, x_labels, k):
    clns_arrived = []
    arrived_features = frozenset()
    all_features = frozenset(x_labels)
    clns_up_to_t, clns_up_to_incl_t = None, None
    while(len(arrived_features) < len(all_features)):
        to_arrive = all_features.difference(arrived_features)
        possible_next = list(get_powerset_to_k_ex_emptyset(to_arrive, k))
        arrives_now_cln = random.choice(possible_next)
        if arrives_now_cln == t_cln:
            clns_up_to_t = copy.deepcopy(clns_arrived)
            clns_up_to_incl_t = copy.deepcopy(clns_arrived) + [arrives_now_cln]
        clns_arrived.append(arrives_now_cln)
        arrived_features = arrived_features.union(arrives_now_cln)
    return clns_up_to_t, clns_up_to_incl_t, clns_arrived


def get_estimate_for_coalition(
    t_cln, 
    num_iterations: int,
    k: int,
    X: pd.DataFrame,
    value_f,
) -> float:
    x_labels = list(X.columns)
    estimates = []
    for itr in range(0, num_iterations):
        rand_seq = random.sample(list(X.index), len(X.index))
        Z = X.loc[rand_seq]
        clns_up_to_t, clns_up_to_incl_t, clns_arrived = get_coalition_arrivals(t_cln, x_labels, k)

        features_arrived = [ft for cln in clns_arrived for ft in cln]

        if t_cln in clns_arrived:
            features_up_to_t = [ft for cln in clns_up_to_t for ft in cln]
            inv_features_up_to_t = [ft for ft in X.columns if ft not in features_up_to_t]
            features_up_to_incl_t = [ft for cln in clns_up_to_incl_t for ft in cln]
            inv_features_up_to_incl_t = [ft for ft in X.columns if ft not in features_up_to_incl_t]
            X_plus_t = pd.concat([
                X.loc[:, features_up_to_incl_t].reset_index(drop=True).astype(np.float64), 
                Z.loc[:, inv_features_up_to_incl_t].reset_index(drop=True).astype(np.float64)
            ], axis=1).loc[:, X.columns]
            X_minus_t = pd.concat([
                X.loc[:, features_up_to_t].reset_index(drop=True).astype(np.float64), 
                Z.loc[:, inv_features_up_to_t].reset_index(drop=True).astype(np.float64)
            ], axis=1).loc[:, X.columns]
            estimates = estimates + [
                (value_f(X_plus_t)) - 
                (value_f(X_minus_t))
            ]
        else:
            estimates = estimates + [np.full(len(X), 0.0)]

    combined_estimates = np.vstack(estimates)
    return combined_estimates.mean(axis=0)


def reduce_to_most_meaningful(local_js, to=10):
    global_js = local_js.abs().mean(axis=0)
    global_js = global_js.sort_values(ascending=False)
    most_meaningful_coalitions = global_js.iloc[:to]
    return most_meaningful_coalitions.index

def calc_attribution(local_js, features):
    attribution_ls = []
    for idx, row in local_js.iterrows():
        temp_dict = {feature: 0.0 for feature in features}
        for index in row.index:
            for f in features:
                if f in index:
                    if isinstance(f, str):
                       temp_dict[f'{f}'] += row[index]
                    else:
                        temp_dict[f] +=row[index]
        attribution_ls.append(np.array(list(temp_dict.values())))
    return np.stack(attribution_ls)



        


if __name__ == '__main__':
    import sklearn
    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")

    X,y = shap.datasets.boston()
    col_rename = {l: f"{l}" for l in X.columns}
    X = X.rename(columns=col_rename)
    y = y

    np.random.seed(0)
    test_train_split = np.random.uniform(0, 1, len(y)) <= .75
    train_X = X[test_train_split]
    train_y = y[test_train_split]
    test_X = X[~test_train_split]
    test_y = y[~test_train_split]

    model = sklearn.ensemble.RandomForestRegressor(n_jobs=10, n_estimators=50,random_state=0).fit(train_X, train_y)
    

    joint_shapley_explainer = Joint_Shapley(model)
    joint_shapley_explainer.explain(test_X)

    new_local = joint_shapley_explainer.interaction
    new_global = np.abs(new_local).mean(axis=0)

    top_local = new_local.loc[:, reduce_to_most_meaningful(new_local)]
    top_global = new_global.loc[reduce_to_most_meaningful(new_local)]

    def show_strip_plot(local_js):
        stacked = local_js.stack().to_frame().rename(columns={0: "Joint Shapley"})
        stacked.loc[:, "Coalition"] = [ind[1] for ind in stacked.index]
        f, ax = plt.subplots()
        sns.despine(bottom=True, left=True)
        ax = sns.stripplot(
            x="Joint Shapley",
            y="Coalition",
            data=stacked, 
            dodge=True, alpha=.5, zorder=1)
        ax.xaxis.label.set_size(10)#set_xlabel("Coalition", fontsize=6)
        plt.show()

    show_strip_plot(top_local)

    f, ax = plt.subplots()
    sns.lineplot(bottom=True, left=True)
    _ = sns.barplot(x=top_global.values, y=top_global.index)
    plt.show()





