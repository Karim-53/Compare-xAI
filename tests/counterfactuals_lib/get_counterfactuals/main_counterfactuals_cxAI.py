from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd
import json
import dice_ml


def get_accuracy(x_test, y_test, model):
    y_pred = model.predict(x_test)
    print("accuracy_score(y_test,y_pred)", accuracy_score(y_test,y_pred))


def get_f1(x_test, y_test, model):
    y_pred = model.predict(x_test)
    print("f1_score(y_test,y_pred)", f1_score(y_test,y_pred))


def create_model(x_train, y_train, svc=True):
    numerical = [
        column
        for column in x_train.columns
        if x_train[column].dtype != 'object' and x_train[column].dtype != 'category'
    ]
    categorical = x_train.columns.difference(numerical)
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    transformations = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical)])
    if svc: clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', SVC(gamma='auto', probability=True))])
    else:   clf = Pipeline(steps=[('preprocessor', transformations), ('classifier', RandomForestClassifier())])
    if categorical.empty:
        clf=SVC(gamma='auto', probability=True)
    model = clf.fit(x_train, y_train)
    return model


def get_train_test_datasets(target_class_name, path_to_dataset):
    dataset = path_to_dataset
    target = dataset[target_class_name]
    train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)

    x_train = train_dataset.drop(target_class_name, axis=1)
    x_test = test_dataset.drop(target_class_name, axis=1)
    return train_dataset, x_test, x_train, y_train, y_test




def get_counterfactuals(train_dataset, x_test, x_train, y_train, y_test, target_class_name, target_class=1, inst=0, list_of_features_to_vary='all', model_is_svc=True, generation_method='genetic', specific_inst=pd.DataFrame(), model_exists="", num_cfs=100):
    # Construct a data object for DiCE. Since continuous and discrete features have different ways of perturbation, we need to specify the names of the continuous features. DiCE also requires the name of the output variable that the ML model will predict.
    numerical = [
        column
        for column in x_train.columns
        if x_train[column].dtype != 'object' and x_train[column].dtype != 'category'
    ]
    d = dice_ml.Data(dataframe=train_dataset, continuous_features=numerical, outcome_name=target_class_name)
    model = create_model(x_train, y_train, svc=model_is_svc)
    m = dice_ml.Model(model=model, backend="sklearn")
    # Using method=random for generating CFs
    exp = dice_ml.Dice(d, m, method=generation_method)
    not_target_class = 0
    if not target_class: not_target_class = 1

    cfs = exp.generate_counterfactuals(specific_inst,
                                    total_CFs=num_cfs,
                                    desired_class=target_class,
                                    features_to_vary=list_of_features_to_vary)

    cfs = cfs.to_json()
    json_cfs_explanation = json.loads(cfs)

    instance = pd.DataFrame(json_cfs_explanation["test_data"][0], columns = json_cfs_explanation["feature_names_including_target"])
    counterfactuals = pd.DataFrame(json_cfs_explanation["cfs_list"][0], columns = json_cfs_explanation["feature_names_including_target"])
    counterfactuals = counterfactuals.where(counterfactuals[target_class_name]==target_class).dropna().reset_index().drop(['index'], axis='columns')

    return instance, counterfactuals, train_dataset


def get_counterfactuals_xAI(path_dataset, inst, target_class_name='label', number_of_cfs=100):
    train_dataset, x_test, x_train, y_train, y_test = get_train_test_datasets(target_class_name, path_dataset)
    _, cfs, _ = get_counterfactuals(train_dataset, x_test, x_train, y_train, y_test, target_class_name, specific_inst=inst, num_cfs=number_of_cfs)
    return cfs