import pandas as pd
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, f1_score
from tests.test_superclass import Test
from sklearn.svm import SVC
from scipy import stats as st


def get_closest_indices(dataframe, instance, mad, k=1):
    """ Takes dataframe and instance, returns list of indices of rows closest to instance in dataframe. """
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
    all_columns = []
    [all_columns.append("cat") if dataframe[column].dtype == 'object' or dataframe[
        column].dtype == 'category' else all_columns.append("cont") for column in dataframe.columns]

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

    index_smallest_distance = dataframe.nsmallest(k, f'total_distance').index.tolist()
    all_columns = [f'{column}_distance'
                   for column in string_cols + numeric_cols + ['categorical', 'continous', 'total']
                   ]
    dataframe.drop(all_columns, inplace=True, axis='columns')
    return index_smallest_distance


def knn_classify(dataframe, samples, to_predict, df_mad):
    """
    dataframe   = training data for knn (cfs+og)
    instance    = instance to be classified
    to_predict  = column that should be predicted
    """

    """ Select class for instance based on knn (to_predict is name of column to be predicted) """
    prediction = []
    numeric_cols = [
        column
        for column in samples.columns
        if samples[column].dtype != 'object' and samples[column].dtype != 'category'
    ]
    mad = get_mad_values(df_mad[numeric_cols])
    for i in range(len(samples.index)):
        instance = samples.iloc[i]
        nearest_neigbors = get_closest_indices(dataframe.drop([to_predict], axis='columns'), instance, mad)
        prediction.append(dataframe.loc[nearest_neigbors[0]][to_predict])  # NOTE: works only for 1-NN
    return prediction


def classify_samples(dataframe_og, samples_og, to_predict, complete_train):
    """ Classify samples using data in training data and return newly classified samples and accuracy compared to true labels. """
    complete_train_copy = complete_train.copy()
    samples = samples_og.copy()
    dataframe = dataframe_og.copy()
    y_test = samples[to_predict]
    samples[to_predict] = knn_classify(dataframe, samples.drop([to_predict], axis='columns'), to_predict,
                                       complete_train_copy)
    samples[to_predict] = samples[to_predict].astype('int32')
    return samples, accuracy_score(y_test, samples[to_predict]), f1_score(y_test, samples[to_predict])


def get_mad_values(df):
    """ Takes a dataframe as input and returns median absolute deviation of columns as list. """
    mad_values = [
        st.median_abs_deviation(df[column].tolist())
        for column in df.columns
    ]
    return mad_values


def get_k_closest(dataframe, instance, target_class_name, target_class, k=100):
    """ Get k closest rows to instance from each class. """
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
    if target_class_name in numeric_cols: numeric_cols.remove(target_class_name)
    if target_class_name in string_cols: string_cols.remove(target_class_name)
    mad = get_mad_values(dataframe[numeric_cols])
    all_columns = []
    [all_columns.append("cat") if dataframe[column].dtype == 'object' or dataframe[
        column].dtype == 'category' else all_columns.append("cont") for column in dataframe.columns]

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

    df_target = dataframe.copy()
    df_target = df_target.loc[df_target[target_class_name] == target_class]
    df_target = df_target.nsmallest(k, f'total_distance')
    df_original = dataframe.copy()
    df_original = df_original.loc[df_original[target_class_name] != target_class]
    df_original = df_original.nsmallest(k, f'total_distance')
    df = pd.concat([df_original, df_target], ignore_index=True)

    all_columns = [f'{column}_distance'
                   for column in string_cols + numeric_cols + ['categorical', 'continous', 'total']
                   ]
    dataframe.drop(all_columns, inplace=True, axis='columns')
    df.drop(all_columns, inplace=True, axis='columns')
    return df


def get_f1_score(counterfactuals, dataset, instance, target_class_name='label', target_class=1, k=100):
    counterfactuals_df = pd.DataFrame(counterfactuals, columns=['x', 'y', target_class_name])
    dataset_copy = dataset.copy()
    dataset_closest = get_k_closest(dataset_copy, instance.iloc[0], target_class_name, target_class, k).reset_index().drop(['index'], axis='columns')
    knn_my_metrics = pd.concat([counterfactuals_df.head(3), instance]).reset_index().drop(['index'], axis='columns')
    _, _, f1_knn = classify_samples(knn_my_metrics, dataset_closest, target_class_name, dataset)
    return f1_knn


def create_data():
    X, y = make_circles(n_samples=1000, random_state=1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df


def create_data_noise():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1, random_state=1)
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    return df


def create_data_imbalanced(t=0):
    X_1, y_1 = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=1)
    df_1 = pd.DataFrame(dict(x=X_1[:, 0], y=X_1[:, 1], label=y_1))
    X_2, y_2 = make_circles(n_samples=1000, noise=0.05, factor=0.5)
    df_2 = pd.DataFrame(dict(x=X_2[:, 0], y=X_2[:, 1], label=y_2))
    df_2 = df_2.drop(df_2[df_2.label == t].index)
    df_3 = pd.concat([df_1, df_2], ignore_index=True)
    return df_3


class Circle(Test):
    name = "1nn_circle"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data()
    trained_model = None
    predict_func = None

    def __init__(self, truth_to_explain=pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleNoise(Test):
    name = "1nn_circle_noise"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_noise()
    trained_model = None
    predict_func = None

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleOutlier(Test):
    name = "1nn_circle_outlier"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_noise()
    trained_model = None
    predict_func = None

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1.5]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_noise()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1.5], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleBigTarget(Test):
    name = "1nn_circle_big_target"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_imbalanced()
    trained_model = None
    predict_func = None

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_imbalanced()
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}


class CircleSmallTarget(Test):
    name = "1nn_circle_small_target"
    ml_task = 'classification'
    input_features = ['x', 'y']
    dataset_size = 1000
    dataset_to_explain = create_data_imbalanced(1)
    trained_model = None
    predict_func = None

    def __init__(self, truth_to_explain= pd.DataFrame({'x': [0], 'y': [-1]}, index=[0]), **kwargs):
        self.dataset_to_explain = create_data_imbalanced(1)
        self.trained_model = SVC(gamma='auto', probability=True)
        x_train = self.dataset_to_explain[['x', 'y']]
        y_train = self.dataset_to_explain['label']
        self.trained_model.fit(x_train, y_train)
        self.predict_func = self.trained_model.predict

        self.truth_to_explain = truth_to_explain

    @classmethod
    def score(cls, instance=pd.DataFrame({'x': [0], 'y': [-1], 'label': [0]}, index=[0]), counterfactual=None, **kwargs): # TODO: how do I tell the score the instance when I only give it the cfs from the explainer?
        return {'1NN_score': get_f1_score(counterfactual, cls.dataset_to_explain, instance)}
