import numpy as np
import pandas as pd
import scipy.stats as st


def bin_continous_values(df, df_target, instance):
    """ Bin all the dataframes to get only categorical values. """
    len_df = len(df.index)
    df_combined = pd.concat([df, df_target, instance], ignore_index=True)
    numeric_cols = [
        column
        for column in df_combined.columns
        if df_combined[column].dtype not in ['object', 'category']
    ]
    for numeric_col in numeric_cols:
        min_value = df_combined[numeric_col].min(axis=0)
        max_value = df_combined[numeric_col].max(axis=0)
        bins = np.linspace(min_value, max_value, NUMBER_OF_BINS)
        df_combined[numeric_col] = pd.cut(
            df_combined[numeric_col], bins=bins, include_lowest=True
        )

    df_binned = df_combined.iloc[:len_df].reset_index().drop(['index'], axis=1)
    df_target_binned = df_combined.iloc[len_df:-1].reset_index().drop(['index'], axis=1)
    instance_binned = df_combined.iloc[[-1]].reset_index().drop(['index'], axis=1)
    return df_binned, df_target_binned, instance_binned


def entropy(column):
    """ Compute entropy of a pandas series. """
    probability_counts = column.value_counts()/sum(column.value_counts().values) # NOTE:might not need to divide by 'sum(column.value_counts().values)'
    return st.entropy(probability_counts)


def generality_score_column(column, column_target, value):
    """ Compute generality score for one feature in the given column. """
    column_with_feature = entropy(column)
    column_without_feature = entropy(column[column != value])
    column_target_with_feature = entropy(column_target)
    column_target_without_feature = entropy(column_target[column_target != value])

    if abs(column_with_feature - column_without_feature) == 0: return 0
    return abs(
        column_target_with_feature - column_target_without_feature
    ) - abs(column_with_feature - column_without_feature)


def generality_score_instance(df, df_target, instance,target_class_name):
    """ Compute generality score for whole instance. """
    df_target = df_target.drop([target_class_name], axis=1)
    df = df.drop([target_class_name], axis=1)
    instance = instance.drop([target_class_name], axis=1)
    df, df_target, instance = bin_continous_values(df, df_target, instance)
    return sum(
        generality_score_column(
            df[column], df_target[column], instance.iloc[0][column]
        )
        for column in df.columns
    )


def get_generality_score(df, df_counterfactuals, target_class, target_class_name) -> pd.DataFrame:
    """ Compute generality score for all counterfactuals. """
    df_target   = df.loc[df[target_class_name] == target_class].reset_index().drop(['index'], axis=1)
    df_counterfactuals_copy = df_counterfactuals.drop(['abnormality'], axis=1)
    scores = []
    for i in range(len(df_counterfactuals_copy.index)):
        instance =  df_counterfactuals_copy.iloc[[i]].reset_index().drop(['index'], axis=1)
        scores.append(generality_score_instance(df,df_target, instance,target_class_name))
    df_counterfactuals['generality'] = scores
    return df_counterfactuals

NUMBER_OF_BINS = 10