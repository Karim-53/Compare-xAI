from scipy import stats as st
import pandas as pd
import prepare_dataset as prep



def get_mad_values(df):
    """ Takes a dataframe as input and returns median absolute deviation of columns as list. """
    mad_values = [
        st.median_abs_deviation(df[column].tolist()) 
        for column in df.columns
    ]
    return mad_values


def get_mean_closest_distance(dataframe, instance, k=3):
    """ Takes dataframe and instance, returns mean of distances of k closest instances in 'dataframe' to the 'instance'. """
    numeric_cols = [
        column
        for column in dataframe.columns
        if dataframe[column].dtype != 'object' and dataframe[column].dtype != 'category'
    ]
    string_cols = [
        column 
        for column in dataframe.columns 
        if dataframe[column].dtype == 'object' or dataframe[column].dtype == 'category'
    ]
    mad = get_mad_values(dataframe[numeric_cols])
    all_columns = []
    [all_columns.append("cat") if dataframe[column].dtype == 'object' or dataframe[column].dtype == 'category' else all_columns.append("cont") for column in dataframe.columns ]

    dataframe[f'categorical_distance'] = 0.
    if string_cols:
        for column in string_cols:
            dataframe[f'{column}_distance'] = (dataframe[f'{column}'] != instance[column]).astype('float')
            dataframe[f'categorical_distance'] = dataframe[f'categorical_distance'] + dataframe[f'{column}_distance']
        dataframe[f'categorical_distance'] = dataframe[f'categorical_distance'] / len(string_cols)
    
    dataframe[f'continous_distance'] = 0.
    if numeric_cols:
        index = 0
        for column in numeric_cols:
            mad[index] = mad[index] if mad[index] != 0 else 1
            dataframe[f'{column}_distance'] = (abs(dataframe[column] - instance[column])) / mad[index]
            dataframe[f'continous_distance'] = dataframe[f'continous_distance'] + dataframe[f'{column}_distance']
            index = index + 1
        dataframe[f'continous_distance'] = dataframe[f'continous_distance'] / len(numeric_cols)

    dataframe[f'total_distance'] = dataframe[f'categorical_distance'] + dataframe[f'continous_distance']
    
    # distance_old = dataframe[f'total_distance'].min()

    distance = dataframe.nsmallest(k, f'total_distance')
    distance = distance[f'total_distance'].mean()
    all_columns = [f'{column}_distance'
                   for column in string_cols + numeric_cols + ['categorical', 'continous','total']
                   ]
    dataframe.drop(all_columns, inplace=True, axis='columns')
    return distance

def get_proximity_score(dataframe, dataframe_of_counterfactuals, target_class, target_class_name) -> pd.DataFrame:
    """ Returns dataframe with additional column 'proximity' containing the distances to the closest instance in 'dataframe' to each instance in 'dataframe_of_counterfactuals'. """
    dataframe_target = dataframe.loc[dataframe[target_class_name] == target_class].reset_index().drop(['index',target_class_name], axis=1)
    dataframe_of_counterfactuals = dataframe_of_counterfactuals.drop([target_class_name], axis=1)
    scores = []
    for i in range(len(dataframe_of_counterfactuals.index)):
        instance =  dataframe_of_counterfactuals.iloc[i]
        counterfactual_proximity = get_mean_closest_distance(dataframe_target,instance)
        scores.append(counterfactual_proximity)
    dataframe_of_counterfactuals['proximity'] = scores
    return dataframe_of_counterfactuals



if __name__ == '__main__':
    
    target_class = '>50K'
    target_class_name = 'income'

    dataframe = prep.prepare_adult_data()
    counterfactuals = dataframe.tail(n=500)
    counterfactuals = counterfactuals.loc[counterfactuals[target_class_name] == target_class].reset_index().drop(['index'], axis=1)
    dataframe = dataframe.head(n=4000)

    print("get_proximity_score(dataframe,counterfactuals)\n", get_proximity_score(dataframe, counterfactuals, target_class, target_class_name))