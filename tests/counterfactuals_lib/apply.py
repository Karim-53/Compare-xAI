import pandas as pd
from tests.counterfactuals_lib import abnormality as abno
from tests.counterfactuals_lib import generality as gene
from tests.counterfactuals_lib import obtainability as obta
from tests.counterfactuals_lib import proximity as prox
from tests.counterfactuals_lib import prepare_dataset as prep
from tests.counterfactuals_lib import moo


def get_random_cfs(df, length, target_class_name, target_class):
    df = df.loc[df[target_class_name] == target_class].reset_index().drop(['index'], axis=1)
    return df.sample(length, replace=True).reset_index().drop(['index'], axis='columns')


def apply_all_metrics(dataframe, instance, dataframe_of_counterfactuals, target_class, target_class_name, ordering=False, metrics=['abnormality', 'generality', 'proximity', 'obtainability']) -> pd.DataFrame:
    """
    Apply all interpretability metrics metrics, returns:
    > dataframe_of_counterfactuals with additional columns representing my metrics
    > newly ordered by pareto optimization
    > random counterfacuals
    """
    dataframe_of_counterfactuals = abno.get_abnormality_score(dataframe, instance, dataframe_of_counterfactuals, target_class, target_class_name)
    dataframe_of_counterfactuals = gene.get_generality_score(dataframe, dataframe_of_counterfactuals, target_class, target_class_name)
    dataframe_of_counterfactuals = prox.get_proximity_score(dataframe, dataframe_of_counterfactuals, target_class, target_class_name)
    dataframe_of_counterfactuals = obta.get_obtainability_score(instance, dataframe_of_counterfactuals, ordering)
    return dataframe_of_counterfactuals, moo.new_order(dataframe_of_counterfactuals, metrics, columns_to_invert=['generality']), get_random_cfs(dataframe, len(dataframe_of_counterfactuals.index), target_class_name, target_class)
