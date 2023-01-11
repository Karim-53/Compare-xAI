import pandas as pd
import numpy as np
from sklearn.datasets import make_circles, make_blobs
# from scipy.io.arff import loadarff
from dice_ml.utils import helpers
from sklearn.utils import resample


def prepare_adult_data() -> pd.DataFrame:
    # TODO: change it to standard dataset
    """ Prepare the adult dataset. """
    # NOTE: some cells have question marks!
    df = pd.read_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adult.csv")
    # df = df.head(10) # NOTE: comment this line out, gets just the first 100 rows just for trying things
    df = df.drop('educational-num', axis=1) # this is unnecessary already have the education level as a categorical value
    df = df.drop('capital-gain', axis=1) # NOTE: MAD is zero (cant divide by zero)
    df = df.drop('capital-loss', axis=1) # NOTE: MAD is zero too
    df = df.drop('fnlwgt', axis=1) # is a weird feature, see "https://www.kaggle.com/datasets/uciml/adult-census-income/discussion/32698"
    # df = df.drop('relationship', axis=1) # TODO: should I drop this column? (you have to create a new model if you drop it!)
    df = df.mask(df.eq('?')).dropna() # delete all rows with missing value which is '?' in the case of the adult data set
    df = df.reset_index(drop=True) # give dataframe new indexes
    return df


def standardize_adult_data():
    df = prepare_adult_data()
    df['income'] = df['income'].replace({'<=50K': 0, '>50K':1})
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\real_world_data\data\census.csv", index=False)


# def balance_adult_data():
#     df = prepare_adult_data()
#     # print("\ndf\n", df)
#     df['income'] = df['income'].replace({'<=50K': 0, '>50K':1})
#     print("\ndf\n", df['income'].value_counts())
#     df_majority = df[df['income']==0]
#     df_minority = df[df['income']==1]
#     df_majority_sampled = resample(df_majority, replace=False, n_samples=11208, random_state=123)
#     df_balanced = pd.concat([df_majority_sampled, df_minority]).reset_index().drop(['index'], axis='columns')
#     df_balanced.to_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adult_balanced.csv", index=False)




def balance_dating_data():
    df = pd.read_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_data_custom.csv")
    print("\ndf\n", df)
    print("\ndf\n", df['match'].value_counts())
    df_majority = df[df['match']==0]
    df_minority = df[df['match']==1]
    df_majority_sampled = resample(df_majority, replace=False, n_samples=1180, random_state=123)
    df_balanced = pd.concat([df_majority_sampled, df_minority]).reset_index().drop(['index'], axis='columns')
    print("\ndf_balanced\n", df_balanced)
    print("\nvalue counts\n", df_balanced['match'].value_counts())
    df_balanced.to_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_balanced.csv", index=False)


def balance_iris_data():
    # TODO
    pass


def prepare_test_data():
    test_data = {'cont': [0, 2.0, 3.4, 5.0, 10.0], 'cat': ['medium', 'good', 'good', 'good', 'bad'], 'target':[0,1,1,1,0]}
    df = pd.DataFrame(data=test_data)

    test_data_instance = {'cont': [5.0], 'cat': ['bad'], 'target':[0]}
    og = pd.DataFrame(data=test_data_instance)
    return df, og


def prepare_test_data_cf():
    test_data = {'cont': [2.0, 3.0, 4.0], 'cat': ['good', 'bad', 'medium'], 'target':[1,1,1]}
    df = pd.DataFrame(data=test_data)
    return df


def prepare_toy_data() -> pd.DataFrame:
    df = pd.read_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\toy_dataset_copy.csv")
    return df


def create_adm_data():
    # https://www.kaggle.com/datasets/akshaydattatraykhare/data-for-admission-in-the-university?resource=download&select=adm_data.csv
    
    """
    This dataset includes various information like GRE score, TOEFL score, university rating, SOP (Statement of Purpose), LOR (Letter of Recommendation), 
    CGPA, research and chance of admit. In this dataset, 400 entries are included.

    GRE Scores ( out of 340 )
    TOEFL Scores ( out of 120 )
    University Rating ( out of 5 )
    Statement of Purpose (SOP) and Letter of Recommendation (LOR) Strength ( out of 5 )
    Undergraduate GPA ( out of 10 )
    Research Experience ( either 0 or 1 )
    Chance of Admit ( ranging from 0 to 1 )
    """

    df = pd.read_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adm_data.csv")
    print("\ncolumns\n", df.columns)
    print("\ndf.info()\n", df.info())
    df = df.drop(["Serial No."], axis='columns')
    print("\ndf.head\n", df.head(20))
    df['Chance of Admit '] = np.where(df['Chance of Admit '] > 0.6, 1, 0) # s.t. about 25% is of class '0'
    df = df.rename({'Serial No.': 'Serial No', 
                    'GRE Score': 'GRE Score', 
                    'TOEFL Score': 'TOEFL Score', 
                    'University Rating': 'University Rating', 
                    'SOP': 'SOP', 
                    'LOR ': 'LOR', 
                    'CGPA': 'CGPA', 
                    'Research': 'Research', 
                    'Chance of Admit ': 'Chance of Admit'}, 
                    axis='columns')
    df['SOP'] = df['SOP'].astype('int32').astype(str)
    df['LOR'] = df['LOR'].astype('int32').astype(str)
    conversion = {'1':'F', '2':'D', '3':'C', '4':'B', '5':'A'} 
    # conversion to five letter marks (it is the strength => 5 is best, 1 is worst) for obtainability (might not keep it that way)
    df['SOP'] = df['SOP'].replace(conversion)
    df['LOR'] = df['LOR'].replace(conversion)
    df['Research'] = np.where(df['Research'] == 1, 'y', 'n')
    print("\ndf.head\n", df.tail(40))
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\real_world_data\data\admission.csv", index=False)


def get_adm_data_custom():
    df = pd.read_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\adm_data_custom.csv")
    return df


def create_dating_data():
    # https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=40536
    # file:///C:/Users/simon/Downloads/speed-dating-data-key.pdf
    # https://www.openml.org/search?type=data&sort=runs&status=active&qualities.NumberOfClasses=%3D_2&id=40536

    df = pd.read_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\Speed Dating Data.csv", encoding='cp1252')
    df = df[['gender', 'age', 'race', 'age_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'amb_o', 'shar_o', 'race_o', 'field_cd', 'match']]
    print("\ndf\n", df)
    df = df.dropna()

    df['gender'] = np.where(df['gender'] == 0, 'female', 'male')
    # print("\ndf\n", df.tail(50))
    df['age'] = df['age'].astype('int32')
    df['age_o'] = df['age_o'].astype('int32')
    df['attr_o'] = df['attr_o'].astype('int32')
    df['sinc_o'] = df['sinc_o'].astype('int32')
    df['intel_o'] = df['intel_o'].astype('int32')
    df['fun_o'] = df['fun_o'].astype('int32')
    df['amb_o'] = df['amb_o'].astype('int32')
    df['shar_o'] = df['shar_o'].astype('int32')
    print("sinco", df['sinc_o'].unique())
    df['attr_o'] = df['attr_o'].replace({
        0: 'slightly_below',
        1: 'slightly_below',
        2: 'below',
        3: 'below',
        4: 'average',
        5: 'average',
        6: 'average',
        7: 'sligthly_above',
        8: 'sligthly_above',
        9: 'above',
        10: 'above'
        })
    
    df['race'] = df['race'].replace({1.0: 'African American', 2.0: 'Caucasian American', 3.0: 'Hispanic American', 4.0:'Asian-American', 5.0: 'Native American', 6.0: 'Other'})
    df['race_o'] = df['race_o'].replace({1.0: 'African American', 2.0: 'Caucasian American', 3.0: 'Hispanic American', 4.0:'Asian-American', 5.0:'Native American', 6.0: 'Other'})
    df['field_cd'] = df['field_cd'].replace({
        1.0: 'Law', 
        2.0: 'Math', 
        3.0: 'Social Science, Psychologist', 
        4.0: 'Medical Science, Pharmaceuticals, Bio Tech',
        5.0: 'Engineering', 
        6.0: 'English/Creative Writing/Journalism', 
        7.0: 'History/Religion/Philosophy', 
        8.0: 'Business/Econ/Finance',
        9.0: 'Education, Academia', 
        10.0: 'Biological Sciences/Chemistry/Physics', 
        11.0: 'Social Work', 
        12.0: 'Undergrad/undecided', 
        13.0: 'Political Science/International Affairs', 
        14.0: 'Film', 
        15.0: 'Fine Arts/Arts Administration', 
        16.0: 'Languages', 
        17.0: 'Architecture', 
        18.0: 'Other'
        })
    
    print("\ndf.columns", df.columns)
    df = df.rename(columns={'field_cd': 'field_study', 'age_o': 'age_partner', 'attr_o': 'attractiveness', 'sinc_o': 'sincerity', 'intel_o': 'intelligence', 'fun_o': 'funny', 'amb_o': 'ambitious', 'shar_o': 'shared_interests', 'race_o': 'race_partner'})
    # TODO: see how balance of data affects my knn accuracy
    # TODO: which columns for obtainability?

    # print("\ninfo\n", df.info())
    print("\ndescribe\n", df.describe())
    print("\ndf\n", df.tail(50))

    df.to_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\dating_data_custom.csv", index=False)


def german_credit():
    # https://towardsdatascience.com/german-credit-data-part-1-exploratory-data-analysis-8f5f266b2426
    # https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk
    df = pd.read_csv("Bachelor-Bench\code\datasets\german_credit_data.csv", index_col=0)
    df["Saving accounts"].fillna("none", inplace=True)
    df["Checking account"].fillna("none", inplace=True)
    # print("\ndf\n", df.columns)
    df['Risk'] = np.where(df['Risk'] == 'good', 1, 0)
    print("\ndf\n", df)
    df['Job'] = df['Job'].replace({
    0: 'unskilled and non-resident', 
    1: 'unskilled and resident', 
    2: 'skilled',
    3: 'highly skilled'
    })

    df.to_csv(r"Bachelor-Bench\code\real_world_data\data\credit.csv", index=False)
    print("\ndf\n", df.head(50))


def adult():
    df = helpers.load_adult_income_dataset()
    print("\ndf\n", df)
    df.to_csv(r"Bachelor-Bench\code\real_world_data\data\adult_dice.csv", index=False)


def iris_data():
    # TODO: make iris dataset (see Bachelor-Bench\code\tryingThings\iris_dataset.py)
    pass


def wine():
    df = pd.read_csv(r"Bachelor-Bench\code\datasets\wine.csv")
    print("\ndfq\n", df)

    df['quality'] = np.where(df['quality'] == 'good', 1, 0)
    print("\ndfq\n", df)
    df.to_csv(r"Bachelor-Bench\code\real_world_data\data\wine.csv", index=False)


def bin_continous_values(df, num_bins=10):
    """ 
    Bin all the dataframes to get only categorical values.
    Helper function 
    """
    numeric_cols = [column
        for column in df.columns 
        if df[column].dtype != 'object' and df[column].dtype != 'category'
    ]
    for i in range(int(len(numeric_cols)/2)):
        min_value = df[numeric_cols[i]].min(axis=0)
        max_value = df[numeric_cols[i]].max(axis=0)
        bins = np.linspace(min_value, max_value, num_bins)
        df[numeric_cols[i]] = pd.cut(df[numeric_cols[i]], bins=bins, include_lowest=True)
    return df


def bins_to_ordinal(df):
    """ Helper function """
    cat_columns = [column
        for column in df.columns 
        if df[column].dtype == 'object' or df[column].dtype == 'category'
    ]

    for column in cat_columns:
        df[column] = df[column].cat.rename_categories(['a','b','c','d','e','f','g','h','i'])


def unit_test_high_number_features():
    feat = 20
    X, y = make_blobs(n_samples=1000, centers=2, n_features=feat, random_state=1, cluster_std=10)

    df = pd.DataFrame()
    for i in range(feat):
        name = 'x_'+str(i)
        df[name] = X[:,i]
    df.insert(feat, 'label', y)

    bin_continous_values(df)
    bins_to_ordinal(df)

    print("\ndf\n", df)
    # df.to_csv(r"\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\datasets\blobs_feats_ut.csv", index=False)

    return df


def unit_test_circles_close():
    X, y = make_circles(n_samples=1000)
    df = pd.DataFrame(dict(x = X[:,0], y = X[:,1], label = y))
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\unit_tests\out\circle_close.csv", index=False)
    return df


def unit_test_imbalanced(t=0):
    X_1, y_1 = make_circles(n_samples=500, noise=0.05, factor=0.5)
    df_1 = pd.DataFrame(dict(x = X_1[:,0], y = X_1[:,1], label = y_1))

    X_2, y_2 = make_circles(n_samples=1000, noise=0.05, factor=0.5)
    df_2 = pd.DataFrame(dict(x = X_2[:,0], y = X_2[:,1], label = y_2))
    df_2 = df_2.drop(df_2[df_2.label==t].index)
    df_3 = pd.concat([df_1, df_2], ignore_index=True)
    # print("\ndf_3\n", df_3['label'].value_counts())
    df_3.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\unit_tests\data\circles_imbalance_Bta_1000.csv", index=False)
    return df_3


def unit_test_circles():
    X, y = make_circles(n_samples=1000)
    df = pd.DataFrame(dict(x = X[:,0], y = X[:,1], label = y))
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\unit_tests\data\ut_circles.csv", index=False)
    return df


def unit_test_circles_noise():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1) # noise=0 or noise=0.1 works okay (works with genetic better than random)
    df = pd.DataFrame(dict(x = X[:,0], y = X[:,1], label = y))
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\unit_tests\data\ut_circles_noise.csv", index=False)
    return df


def unit_test_circles_outlier():
    X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1) # noise=0 or noise=0.1 works okay (works with genetic better than random)
    df = pd.DataFrame(dict(x = X[:,0], y = X[:,1], label = y))
    df.to_csv(r"C:\Users\simon\OneDrive\Desktop\thesis\Bachelor-Bench\code\unit_tests\data\circles_outlier_noise.csv", index=False)
    return df


if __name__ == '__main__':
    unit_test_high_number_features()