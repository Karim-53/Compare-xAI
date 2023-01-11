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


if __name__ == '__main__':

    # ordering of ordinal features in dataset, from low to high
    ordering_features_adult_data = {
        "education": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors',  'Masters', 'Doctorate']
    }

    ordering_features_toy_data = {
        "size": ['10', '20', '30', '40', '50']
    }

    # target_class_name   = 'label'
    # target_class        = 1

    # df      = prep.prepare_toy_data()
    # cf      = df.tail(n=20)
    # cf      = cf.loc[cf[target_class_name] == target_class].reset_index().drop(['index'], axis=1)
    # inst    = df.iloc[[470]].reset_index().drop(['index'], axis=1)
    # df      = df.head(n=450)

    # print("cf     \n", cf  )
    # print("inst   \n", inst)
    # print("df     \n", df  )
    # print("", )

    # cf = apply_all_metrics(df,inst,cf,target_class,target_class_name,ordering_features_toy_data)

    # print("cf\n", cf)


    target_class = '>50K'
    target_class_name = 'income'

    dataframe = prep.prepare_adult_data()

    counterfactuals = dataframe.tail(n=200)
    counterfactuals = counterfactuals.loc[counterfactuals[target_class_name] == target_class].reset_index().drop(['index'], axis=1)

    # instance = dataframe.iloc[[40005]].reset_index().drop(['index'], axis=1) # no abnormal values
    instance = dataframe.iloc[[40724]].reset_index().drop(['index'], axis=1) # age = 17 is abnormal

    dataframe = dataframe.head(n=40000)

    print("\ndataframe\n", dataframe)

    print("\ninstance\n", instance)
    print("\n\n\n", )

    counterfactuals, cfs_pareto, random = apply_all_metrics(dataframe, instance, counterfactuals, target_class, target_class_name, ordering_features_adult_data, metrics=['abnormality'])
    print("\ncounterfactuals\n", counterfactuals)
    print("\n\n>>> cfs pareto\n", cfs_pareto)
    print("\nrandom\n", random)

    # print("\n\n\n>>> counterfactuals.sort_values(by=['generality']) \n\n", counterfactuals.sort_values(by=['generality']))
    # print("\n\n\n>>> counterfactuals.sort_values(by=['proximity']) \n\n", counterfactuals.sort_values(by=['proximity']))
    # print("\n\n\n>>> counterfactuals.sort_values(by=['obtainability']) \n\n", counterfactuals.sort_values(by=['obtainability']))