# file for obtainability
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

# ordinal_features_adult_data = {
#     "education": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors',  'Masters', 'Doctorate'],
#     # "occupation": ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving']
# }

# df = prep.prepare_adult_data()

# target_class = '>50K'
# target_class_name = 'income'

# counterfactuals = df.tail(n=50)
# counterfactuals = counterfactuals.loc[counterfactuals[target_class_name] == target_class].reset_index().drop(['index'], axis=1)

# instance = df.iloc[[4005]].reset_index().drop(['index'], axis=1)
# df = df.head(n=400)

# print("\n\nget_obtainability_score\n", get_obtainability_score(instance, counterfactuals, ordinal_features_adult_data))

# # print("obtainability(instance, counterfactual, ordinal_features_adult_data)\n", obtainability(instance, counterfactual, ordinal_features_adult_data))

