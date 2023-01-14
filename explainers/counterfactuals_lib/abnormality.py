import pandas as pd
import numpy as np
from explainers.counterfactuals_lib import prepare_dataset as prep


def compute_abnormality_column(some_series, dict_feature_occurrence, percentage=0):
    """
    compute the abnormality for a whole series' cells.
    """
    assert 0 not in list(some_series.map(dict_feature_occurrence).astype(float)), "Some occurence is 0."
    scores = percentage/some_series.map(dict_feature_occurrence).astype(float)    
    return scores


def additive_smoothing(ser, a=1):
    # https://en.wikipedia.org/wiki/Additive_smoothing
    d = len(ser)
    N = ser.sum()
    ser_smooth = ser.apply(lambda x: (x+a)/(N+a*d)*N) # smoothed count
    ser_smooth = ser_smooth.apply(lambda p: p/N) # get probabilities
    return ser_smooth


def get_feature_occurrence(dataframe, columns):
    """ Returns a dictionary containing every feature and its prevelance in the target class in percent. """
    feature_occurrence = {}
    for i in range(len(columns)):
        feature_occurrence[columns[i]] = {}
        dataframe_makeup = dataframe[columns[i]].value_counts()
        feature_occurrence[columns[i]] = additive_smoothing(dataframe_makeup).to_dict()
    return feature_occurrence


def check_for_abnormality(dataframe, dataframe_of_counterfactuals, instance):
    """ Check for abnormality in categorical values. """
    string_cols = [column for column in dataframe.columns if dataframe[column].dtype == 'object' or dataframe[column].dtype == 'category']
    dict_feature_occurrence = get_feature_occurrence(dataframe, string_cols)
    dataframe_of_counterfactuals['abnormality'] = 0
    for column in string_cols:
        feature_value = instance.iloc[0][column]

        if feature_value in dict_feature_occurrence[column] and dict_feature_occurrence[column][feature_value] <= ABNORMAL_THRESHOLD:           
            dataframe_of_counterfactuals['abnormality'] = dataframe_of_counterfactuals['abnormality'] + compute_abnormality_column(dataframe_of_counterfactuals[column], dict_feature_occurrence[column], dict_feature_occurrence[column][feature_value])
        elif feature_value not in dict_feature_occurrence[column]:
            dataframe_of_counterfactuals['abnormality'] = dataframe_of_counterfactuals['abnormality'] + compute_abnormality_column(dataframe_of_counterfactuals[column], dict_feature_occurrence[column])
    return dataframe_of_counterfactuals['abnormality']


def bin_continous_values(df, df_1, instance):
    """ Bin all the dataframes to get only categorical values. """
    len_df = len(df.index)
    df_combined = pd.concat([df, df_1, instance], ignore_index=True)
    numeric_cols = [column
        for column in df_combined.columns 
        if df_combined[column].dtype != 'object' and df_combined[column].dtype != 'category'
    ]
    for i in range(len(numeric_cols)):
        min_value = df_combined[numeric_cols[i]].min(axis=0)
        max_value = df_combined[numeric_cols[i]].max(axis=0)
        bins = np.linspace(min_value, max_value, NUMBER_OF_BINS)
        df_combined[numeric_cols[i]] = pd.cut(df_combined[numeric_cols[i]], bins=bins, include_lowest=True)
    df_binned = df_combined.iloc[:len_df].reset_index().drop(['index'], axis=1)
    df_1_binned = df_combined.iloc[len_df:-1].reset_index().drop(['index'], axis=1)
    instance_binned = df_combined.iloc[[-1]].reset_index().drop(['index'], axis=1)
    return df_binned, df_1_binned, instance_binned


def make_object_category(df, cfs, inst):
    obj_columns = [column
        for column in df.columns 
        if df[column].dtype == 'object'
    ]
    for col in obj_columns:
        df[col] = df[col].astype('category')
        cfs[col] = cfs[col].astype('category').cat.set_categories(df[col].cat.categories)
        inst[col] = inst[col].astype('category').cat.set_categories(df[col].cat.categories)


def get_abnormality_score(dataframe, instance, dataframe_of_counterfactuals, target_class, target_class_name) -> pd.DataFrame:
    """ 
    Apply the metric abnormality:
    > see if abnormal values exist in the instance
    > compute the difference of occurrence in target class of abnormal value and value in cfs
    > sum of these differences is abnormality score
    """
    dataframe_copy = dataframe
    instance_copy = instance.drop(target_class_name, axis=1)
    dataframe_of_counterfactuals_copy = dataframe_of_counterfactuals.drop(target_class_name, axis=1)
    make_object_category(dataframe_copy, dataframe_of_counterfactuals_copy, instance_copy)

    dataframe_copy = dataframe_copy.loc[dataframe_copy[target_class_name] == target_class].drop(target_class_name, axis='columns')
    dataframe_copy, dataframe_of_counterfactuals_copy, instance_copy = bin_continous_values(dataframe_copy, dataframe_of_counterfactuals_copy, instance_copy)
    dataframe_of_counterfactuals['abnormality'] = check_for_abnormality(dataframe_copy, dataframe_of_counterfactuals_copy, instance_copy)
    return dataframe_of_counterfactuals


NUMBER_OF_BINS      = 10
ABNORMAL_THRESHOLD  = 0.05
