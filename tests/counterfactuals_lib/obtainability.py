import pandas as pd

def obtainability(instance, counterfactual, ordinal_dict):
    instance = instance.reset_index().drop(['index'], axis=1)
    counterfactual = counterfactual.reset_index().drop(['index'], axis=1)
    score = 0
    for feature in instance.columns:
        if feature in ordinal_dict:
            index_instance = ordinal_dict[feature].index(str(instance[feature][0]))
            index_counterfactual = ordinal_dict[feature].index(str(counterfactual[feature][0]))
            score = score + abs(index_instance - index_counterfactual)

    return score

def get_obtainability_score(instance, df_counterfactuals, ordinal_dict) -> pd.DataFrame:
    """  Compute obtainability score for all counterfactuals. """
    if not ordinal_dict: 
        df_counterfactuals['obtainability'] = 0
        return df_counterfactuals
    scores = []
    for i in range(len(df_counterfactuals.index)):
        counterfactual =  df_counterfactuals.iloc[[i]]
        scores.append(obtainability(instance, counterfactual, ordinal_dict))
    df_counterfactuals['obtainability'] = scores
    return df_counterfactuals