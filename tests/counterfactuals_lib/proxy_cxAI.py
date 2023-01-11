from statistics import mean, median, stdev
from scipy.stats import sem
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import get_counterfactuals.main_generic as main
# import get_counterfactuals.main_census as census
import proximity as prox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import moo
import apply
import json


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
    mad = prox.get_mad_values(dataframe[numeric_cols])
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
    mad = prox.get_mad_values(df_mad[numeric_cols])
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
    # print("\nclassify_samples\n", )
    # print("\ny_test, samples[to_predict]\n", y_test.tail(50), "\n" , samples[to_predict].tail(50))
    # print("\nrecall\n", recall_score(y_test, samples[to_predict]))
    # print("\nprecision\n", precision_score(y_test, samples[to_predict]))
    return samples, accuracy_score(y_test, samples[to_predict]), f1_score(y_test, samples[to_predict])


def print_readable(number_of_cfs, f1_score_abno, f1_score_gene, f1_score_prox, f1_score_obta, f1_score_all,
                   f1_score_dice, f1_score_rand, f1_score_dice_3):
    """ Helper function. """
    print("\n\nf1_score_all ", f1_score_all)
    print("f1_score_dice", f1_score_dice)
    print("f1_score_dice_3", f1_score_dice_3)
    print("f1_score_rand", f1_score_rand)
    print("f1_score_abno", f1_score_abno)
    print("f1_score_gene", f1_score_gene)
    print("f1_score_prox", f1_score_prox)
    print("f1_score_obta", f1_score_obta)

    print("\n\nf1_score_all  mean", mean(f1_score_all))
    print("f1_score_dice mean", mean(f1_score_dice))
    print("f1_score_dice_3 mean", mean(f1_score_dice_3))
    print("f1_score_rand mean", mean(f1_score_rand))
    print("f1_score_abno mean", mean(f1_score_abno))
    print("f1_score_gene mean", mean(f1_score_gene))
    print("f1_score_prox mean", mean(f1_score_prox))
    print("f1_score_obta mean", mean(f1_score_obta))

    print("\n\nf1_score_all  sem", sem(f1_score_all))
    print("f1_score_dice sem", sem(f1_score_dice))
    print("f1_score_dice_3 sem", sem(f1_score_dice_3))
    print("f1_score_rand sem", sem(f1_score_rand))
    print("f1_score_abno sem", sem(f1_score_abno))
    print("f1_score_gene sem", sem(f1_score_gene))
    print("f1_score_prox sem", sem(f1_score_prox))
    print("f1_score_obta sem", sem(f1_score_obta))

    print("\n\nnumber_of_cfs\t\t", number_of_cfs)
    print("mean number_of_cfs\t", number_of_cfs)


def create_dict(f1_abno, f1_gene, f1_prox, f1_obta, f1_all, f1_dice, f1_rand, f1_dice_3):
    all = dict(f1_mean=mean(f1_all), f1_sem=sem(f1_all))
    dice = dict(f1_mean=mean(f1_dice), f1_sem=sem(f1_dice))
    dice_3 = dict(f1_mean=mean(f1_dice_3), f1_sem=sem(f1_dice_3))
    abno = dict(f1_mean=mean(f1_abno), f1_sem=sem(f1_abno))
    gene = dict(f1_mean=mean(f1_gene), f1_sem=sem(f1_gene))
    prox = dict(f1_mean=mean(f1_prox), f1_sem=sem(f1_prox))
    obta = dict(f1_mean=mean(f1_obta), f1_sem=sem(f1_obta))
    rand = dict(f1_mean=mean(f1_rand), f1_sem=sem(f1_rand))
    complete = {
        'dice': dice,
        'dice_3': dice_3,
        'rand': rand,
        'all': all,
        'abno': abno,
        'gene': gene,
        'prox': prox,
        'obta': obta
    }
    return complete


def prepare_individual_benchmarking(df, sort_by, metrics, target_class_name, target_class, asc=False):
    df_sorted = df.sort_values(by=sort_by, ascending=asc)
    df_sorted = df_sorted.drop(metrics, axis='columns')
    df_sorted[target_class_name] = target_class
    return df_sorted


def do_individual_benchmarking(df, inst, numb_cfs, train_df, target_class_name, list_with_acc, list_with_f1):
    """ For knn with each counterfactuals and instance separately and then mean of each accuracy. """
    list_tmp = []
    for i in numb_cfs:
        if i > len(numb_cfs): break
        knn_my_metrics_abno = pd.concat([df.iloc[[i]], inst]).reset_index().drop(['index'], axis='columns')
        _, accuracy_my_abno, f1_knn = classify_samples(knn_my_metrics_abno, train_df, target_class_name)
        list_tmp.append(accuracy_my_abno)
    list_with_acc.append(mean(list_tmp))
    list_with_f1.append(f1_knn)


def do_counterfactuals_at_once(df, inst, train_df, target_class_name, list_with_f1, complete_train, tmp):
    """ For knn with counterfactuals all at once and instance. """
    knn_my_metrics = pd.concat([df.head(3), inst]).reset_index().drop(['index'], axis='columns')
    _, _, f1_knn = classify_samples(knn_my_metrics, train_df, target_class_name, complete_train)
    list_with_f1.append(f1_knn)
    knn_copy = knn_my_metrics.copy()
    knn_copy['type'] = 'knn'
    train_copy = train_df.copy()
    train_copy['type'] = 'train'

    # combined = pd.concat([train_copy, knn_copy]).reset_index()
    # sns.scatterplot(x='x', y='y', data=combined, hue='type').set(title=tmp)
    # plt.plot()
    # plt.show()


def get_new_row(name_of_dataset, features_varied, metrics_used, f1, number_cfs):
    return {'dataset': name_of_dataset, 'features_varied': features_varied, 'metrics': metrics_used, 'f1_score': f1,
            'numb_of_cfs': number_cfs}


def append_rows_metric(f1, number_of_cfs, name_of_dataset, features_varied, df, metrics_used):
    for i in range(len(f1)):
        row_dict = get_new_row(name_of_dataset, features_varied, metrics_used, f1[i], number_of_cfs[i])
        row_df = pd.DataFrame(row_dict, index=[0])
        df = pd.concat([df, row_df], axis=0, ignore_index=True)
    return df


def output_to_csv(number_of_cfs, name_of_dataset, features_varied, f1_abno, f1_gene, f1_prox, f1_obta, f1_all, f1_dice,
                  f1_rand, f1_dice_3, folder_out=r"Bachelor-Bench\code\real_world_data\out"):
    """ Output accuracies to a csv file for evaluation. """
    names = ['dataset', 'features_varied', 'metrics', 'numb_of_cfs']
    path = folder_out + '\\' + name_of_dataset + '.csv'
    try:
        df = pd.read_csv(path)
    except:
        df = pd.DataFrame(columns=names)

    df = append_rows_metric(f1_dice, number_of_cfs, name_of_dataset, features_varied, df, 'dice')
    df = append_rows_metric(f1_dice_3, number_of_cfs, name_of_dataset, features_varied, df, 'dice_3')
    df = append_rows_metric(f1_rand, number_of_cfs, name_of_dataset, features_varied, df, 'rand')
    df = append_rows_metric(f1_abno, number_of_cfs, name_of_dataset, features_varied, df, 'abno')
    df = append_rows_metric(f1_gene, number_of_cfs, name_of_dataset, features_varied, df, 'gene')
    df = append_rows_metric(f1_prox, number_of_cfs, name_of_dataset, features_varied, df, 'prox')
    df = append_rows_metric(f1_obta, number_of_cfs, name_of_dataset, features_varied, df, 'obta')
    df = append_rows_metric(f1_all, number_of_cfs, name_of_dataset, features_varied, df, 'all')
    df.to_csv(path, index=False)
    return path


def plot_train_inst_cfs(train_dataset, instance, counterfactuals, plot_title):
    """ Plotting for debugging. """
    cf_copy = counterfactuals.copy()
    instance_copy = instance.copy()
    cf_copy['type'] = 'cfs'
    cf_copy['style'] = 'knn'
    instance_copy['type'] = 'inst'
    instance_copy['style'] = 'knn'
    train_copy = train_dataset.copy()
    train_copy['type'] = np.where(train_copy['label'] == 0, 'class:0', 'class:1')
    train_copy['style'] = 'train'

    combined = pd.concat([train_copy, instance_copy, cf_copy]).reset_index()
    sns.scatterplot(x='x', y='y', data=combined, hue='type', style='style', palette='bright').set(title=plot_title)
    plt.plot()
    plt.show()

    return combined


def plot_train_inst_cfs_ax(train_dataset, instance, counterfactuals, plot_title, ax):
    """ Plotting for debugging. """
    cf_copy = counterfactuals.copy()
    instance_copy = instance.copy()
    cf_copy['type'] = 'cfs'
    cf_copy['style'] = 'knn'
    instance_copy['type'] = 'inst'
    instance_copy['style'] = 'knn'
    train_copy = train_dataset.copy()
    train_copy['type'] = np.where(train_copy['label'] == 0, 'class:0', 'class:1')
    train_copy['style'] = 'train'
    combined = pd.concat([train_copy, instance_copy, cf_copy]).reset_index()
    sns.scatterplot(x='x', y='y', data=combined, hue='type', style='style', palette='bright', ax=ax).set(
        title=plot_title)


def rounds_of_classifying(target_class, target_class_name, path, dataset_name, ordering=False, reps=10, rounds=7,
                          to_vary='all', metrics=['abnormality', 'generality', 'proximity', 'obtainability'],
                          numb_cfs=3, svc=True, csv=True, k=100, inst=pd.DataFrame(), model="", out_folder=False):
    """ Classify counterfactuals of datasets with knn. """
    number_of_cfs = []

    f1_score_all = []
    f1_score_rand = []
    f1_score_dice = []
    f1_score_abno = []
    f1_score_gene = []
    f1_score_prox = []
    f1_score_obta = []
    f1_score_dice_3 = []

    for rep in range(reps):
        for r in range(rounds):
            print("\nnew round: ", r)
            print("rep: ", rep)
            train_dataset, x_test, x_train, y_train, y_test = main.get_train_test_datasets(target_class_name, path)

            try:
                instance, counterfactuals, train_dataset = main.get_counterfactuals(train_dataset, x_test, x_train,
                                                                                    y_train, y_test, target_class_name,
                                                                                    target_class=target_class, inst=r,
                                                                                    list_of_features_to_vary=to_vary,
                                                                                    model_is_svc=svc,
                                                                                    specific_inst=inst,
                                                                                    model_exists=model)
                instance_3, counterfactuals_3, train_dataset_3 = main.get_counterfactuals(train_dataset, x_test,
                                                                                          x_train, y_train, y_test,
                                                                                          target_class_name,
                                                                                          target_class=target_class,
                                                                                          inst=r,
                                                                                          list_of_features_to_vary=to_vary,
                                                                                          model_is_svc=svc,
                                                                                          specific_inst=inst,
                                                                                          model_exists=model, num_cfs=3)

                assert instance.equals(instance_3), "Instances are not equal!"
                assert train_dataset.equals(train_dataset_3), "Train Datasets are not equal!"
            except:
                print("could not find counterfactuals, try next one", )
            else:
                number_of_cfs.append(len(counterfactuals))
                _, cfs_pareto, random = apply.apply_all_metrics(train_dataset, instance, counterfactuals, target_class,
                                                                target_class_name, ordering)
                counterfactuals = counterfactuals.drop(['abnormality', 'generality'],
                                                       axis='columns')  # NOTE: put this line in apply.py instead

                cfs_pareto_abno = prepare_individual_benchmarking(cfs_pareto, ['abnormality'], metrics,
                                                                  target_class_name, target_class, asc=True)
                cfs_pareto_gene = prepare_individual_benchmarking(cfs_pareto, ['generality'], metrics,
                                                                  target_class_name, target_class)
                cfs_pareto_prox = prepare_individual_benchmarking(cfs_pareto, ['proximity'], metrics, target_class_name,
                                                                  target_class, asc=True)
                cfs_pareto_obta = prepare_individual_benchmarking(cfs_pareto, ['obtainability'], metrics,
                                                                  target_class_name, target_class, asc=True)
                cfs_pareto = cfs_pareto.drop(metrics, axis='columns')
                cfs_pareto[target_class_name] = target_class
                train_dataset_closest = get_k_closest(train_dataset, instance.iloc[0], target_class_name, target_class,
                                                      k).reset_index().drop(['index'], axis='columns')

                do_counterfactuals_at_once(cfs_pareto_abno, instance, train_dataset_closest, target_class_name,
                                           f1_score_abno, train_dataset, "pareto_abno")
                do_counterfactuals_at_once(cfs_pareto_gene, instance, train_dataset_closest, target_class_name,
                                           f1_score_gene, train_dataset, "pareto_gene")
                do_counterfactuals_at_once(cfs_pareto_prox, instance, train_dataset_closest, target_class_name,
                                           f1_score_prox, train_dataset, "pareto_prox")
                do_counterfactuals_at_once(cfs_pareto_obta, instance, train_dataset_closest, target_class_name,
                                           f1_score_obta, train_dataset, "pareto_obta")
                do_counterfactuals_at_once(cfs_pareto, instance, train_dataset_closest, target_class_name, f1_score_all,
                                           train_dataset, "cfs_pareto")
                do_counterfactuals_at_once(counterfactuals, instance, train_dataset_closest, target_class_name,
                                           f1_score_dice, train_dataset, "cfs_dice")
                do_counterfactuals_at_once(random, instance, train_dataset_closest, target_class_name, f1_score_rand,
                                           train_dataset, "cfs_rand")
                do_counterfactuals_at_once(counterfactuals_3, instance, train_dataset_closest, target_class_name,
                                           f1_score_dice_3, train_dataset, "cfs_dice_3")
    try:
        print_readable(number_of_cfs, f1_score_abno, f1_score_gene, f1_score_prox, f1_score_obta, f1_score_all,
                       f1_score_dice, f1_score_rand, f1_score_dice_3)
    except:
        print("\nno counterfactuals found\n", )

    if to_vary == 'all':
        features_varied = to_vary
    else:
        features_varied = ','.join(to_vary)

    if csv:
        path_to_csv = output_to_csv(number_of_cfs, dataset_name, features_varied, f1_score_abno, f1_score_gene,
                                    f1_score_prox, f1_score_obta, f1_score_all, f1_score_dice, f1_score_rand,
                                    f1_score_dice_3, out_folder)
    else:
        path_to_csv = []

    return path_to_csv, create_dict(f1_score_abno, f1_score_gene, f1_score_prox, f1_score_obta, f1_score_all, f1_score_dice, f1_score_rand, f1_score_dice_3)


def get_f1_score(counterfactuals, dataset, instance, target_class_name='label', target_class=1, k=100):
    dataset_copy = dataset.copy()
    dataset_closest = get_k_closest(dataset_copy, instance.iloc[0], target_class_name, target_class, k).reset_index().drop(['index'], axis='columns')

    knn_my_metrics = pd.concat([counterfactuals.head(3), instance]).reset_index().drop(['index'], axis='columns')
    _, _, f1_knn = classify_samples(knn_my_metrics, dataset_closest, target_class_name, dataset)

    return f1_knn

if __name__ == '__main__':
    # TODO: try out new standardized adult dataset
    # TODO: see how balanced datasets perform

    # print("\nADM\n", )
    # target_class = 1
    # target_class_name = 'Chance of Admit'
    # name_adm = 'Admission_tmp_ind'
    # path_adm = r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adm_data_custom.csv"
    # ordering_features_adm = {
    #     'SOP': ['A', 'B', 'C', 'D', 'F'],
    #     'LOR': ['A', 'B', 'C', 'D', 'F']
    # }
    # to_vary = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR',  'CGPA', 'Research']

    # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=1, rounds=7, to_vary=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR',  'CGPA', 'Research'])  # all
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['SOP', 'LOR'])                                                                        # just obtainability
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['GRE Score', 'TOEFL Score', 'University Rating', 'CGPA', 'Research'])                 # no obtainability
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['GRE Score', 'TOEFL Score',  'CGPA'])                                                 # only real numerical features
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['GRE Score', 'TOEFL Score', 'SOP', 'LOR'])                                            # same numer numerical as continous
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR',  'CGPA'])              # no particular reason
    # # rounds_of_classifying(target_class, target_class_name, path_adm, name_adm, ordering_features_adm, reps=10, rounds=7, to_vary=['GRE Score', 'TOEFL Score', 'SOP', 'LOR',  'CGPA', 'Research'])                       # no particular reason

    ####################################################################################################################
    ####################################################################################################################

    # print("\nDATING\n", )

    # target_class_name = 'match'
    # target_class = 1
    # path_dating = r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_data_custom.csv"
    # ordering_features_dating = {
    #     'attractiveness': ['slightly_below', 'below', 'average', 'sligthly_above', 'above']
    # }
    # # to_vary = ['gender', 'age', 'race', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study']

    # rounds_of_classifying(target_class, target_class_name, path_dating, ordering_features_dating, reps=1, to_vary='all')

    ####################################################################################################################
    ####################################################################################################################

    # print("\nADULT\n", )
    # ordering_features_adult_data = {
    #     "education": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors',  'Masters', 'Doctorate']
    # }
    # # to_vary = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'hours-per-week', 'native-country']
    # target_class_name_adult = 'income'
    # target_class_adult = 1
    # adult_name = 'adult_test'
    # path_adult = r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adult_standard.csv"
    # rounds_of_classifying(target_class_adult, target_class_name_adult, path_adult, adult_name, ordering_features_adult_data, reps=1, rounds=1, svc=False)

    ######################################################################################################################
    ######################################################################################################################

    print("\nIRIS\n", )

    target_class_name = 'label'
    iris_name = 'Test_tmp_ind'
    target_class = 1
    path_iris = r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\toy_dataset_iris_3d.csv"
    ordering_features_iris_3d = {"petal length": ['v', 's', 'p']}
    # st = time.time()
    # to_vary = ['sepal length', 'sepal width', 'petal length']

    print(rounds_of_classifying(target_class, target_class_name, path_iris, iris_name, ordering_features_iris_3d,
                                to_vary=['sepal length', 'sepal width', 'petal length'], reps=1, rounds=2, svc=False,
                                k=10))
    # rounds_of_classifying(target_class, target_class_name, path_iris, iris_name, ordering_features_iris_3d, to_vary=['sepal length', 'petal length'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_iris, iris_name, ordering_features_iris_3d, to_vary=['sepal width', 'petal length'], reps=10, rounds=10, svc=False)

    # et = time.time()
    # elapsed_time = et - st
    # print('\nExecution time:', elapsed_time, 'seconds')

    ######################################################################################################################
    ######################################################################################################################

    # print("\nDATING\n", )
    # dating_name = 'Dating'
    # target_class_name = 'match'
    # target_class = 1
    # path_dating = r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_data_custom.csv"
    # ordering_features_dating = {
    #     'attractiveness': ['slightly_below', 'below', 'average', 'sligthly_above', 'above']
    # }
    # # to_vary = ['gender', 'age', 'race', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study']
    # # fix dating maybe try balanced
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['gender', 'age', 'race', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study'], reps=1, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['gender', 'age', 'race', 'age_partner', 'attractiveness', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['age', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['gender', 'race', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['gender', 'age', 'race', 'age_partner', 'funny', 'ambitious', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['age', 'age_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating, to_vary=['attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)

    # print("\nDATING Balanced\n", )
    # dating_name = 'Dating balanced'
    # target_class_name = 'match'
    # target_class = 1
    # path_dating = r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_balanced.csv"
    # ordering_features_dating = {
    #     'attractiveness': ['slightly_below', 'below', 'average', 'sligthly_above', 'above']
    # }
    # # to_vary = ['gender', 'age', 'race', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study']
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['gender', 'age', 'race', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['gender', 'age', 'race', 'age_partner', 'attractiveness', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['age', 'age_partner', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests'], reps=10, rounds=10, svc=False)
    # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['gender', 'race', 'attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['gender', 'age', 'race', 'age_partner', 'funny', 'ambitious', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['age', 'age_partner', 'field_study'], reps=10, rounds=10, svc=False)
    # # rounds_of_classifying(target_class, target_class_name, path_dating, dating_name, ordering_features_dating,  to_vary=['attractiveness', 'sincerity', 'intelligence', 'funny', 'ambitious', 'shared_interests', 'race_partner', 'field_study'], reps=10, rounds=10, svc=False)