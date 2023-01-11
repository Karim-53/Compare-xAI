import proxy
import os
import pandas as pd

class Unit_Tests():
    name = 'Unit Tests'
    ml_task = 'classification'
    table_path = r"Bachelor-Bench\code\unit_tests\unit_test_table.csv"

    def __init__(self, target_class, target_class_name, name, path, ordering_features=False, knn_size=100, specific_inst=pd.DataFrame(), out=r"Bachelor-Bench\code\unit_tests\out", rounds=50, reps=1):
        self.target_class = target_class
        self.target_class_name = target_class_name
        self.name = name
        self.path = path
        self.ordering_features = ordering_features
        # self.to_vary = to_vary
        self.scores = {}
        self.csv_file = ""
        self.knn_size = knn_size
        self.specific_inst = specific_inst
        self.out = out
        self.rounds = rounds
        self.reps = reps

    def create_new_score(self):
        self.csv_file, self.scores = proxy.rounds_of_classifying(self.target_class, self.target_class_name, self.path, self.name, self.ordering_features, reps=self.reps, rounds=self.rounds, k=self.knn_size, inst=self.specific_inst, out_folder=self.out)  # CHANGE: 
        print("\nscores\n", self.scores)
    
    def print_scores(self):
        pass

    def get_existing_scores(self):
        path = r"Bachelor-Bench\code\unit_tests\out" + '\\' + self.name + '.csv'
        isFile = os.path.isfile(path)
        if isFile:
            self.csv_file = path 
            df = pd.read_csv(path)
            df = df.groupby(by='metrics').f1_score.agg(['mean','std']).rename(columns={'mean':'f1_mean', 'std':'f1_stdev'})
            df = df.reset_index().set_index(['metrics'])
            self.scores = df.to_dict(orient='index')
    
    def scores_to_table(self, path):
        df = pd.DataFrame(self.scores)
        df = df.round(4)
        df = df.reset_index().drop(['index'], axis='columns')
        for column in df.columns:
            df[column] = df[column].astype('str')
            df.at[0,column] = ', '.join(df[column].astype('str'))
        df = df.drop([1])
        df.insert(0, 'unit test', self.name)
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))


class Evaluation():
    name = 'Evaluation'
    ml_task = 'classification'
    table_path = r"Bachelor-Bench\code\real_world_data\real_world_table.csv"

    def __init__(self, target_class, target_class_name, name, path, ordering_features=False, knn_size=100, specific_inst=pd.DataFrame(), to_vary='all', svc=True, model="", rounds=10, reps=5, out=r"Bachelor-Bench\code\real_world_data\out"):
        self.target_class = target_class
        self.target_class_name = target_class_name
        self.name = name
        self.path = path
        self.ordering_features = ordering_features
        self.to_vary = to_vary
        self.scores = {}
        self.svc = svc
        self.csv_file = ""
        self.model = model
        self.knn_size = knn_size
        self.specific_inst = specific_inst
        self.rounds = rounds
        self.reps = reps
        self.out= out

    def create_new_score(self):
        self.csv_file, self.scores = proxy.rounds_of_classifying(self.target_class, self.target_class_name, self.path, self.name, self.ordering_features, reps=self.reps, rounds=self.rounds, k=self.knn_size, inst=self.specific_inst, to_vary=self.to_vary, svc=self.svc, model=self.model, out_folder=self.out)
        print("\nscores\n", self.scores)
    
    def print_scores(self):
        pass

    def get_existing_scores(self): # TODO: change it to sem
        path = r"Bachelor-Bench\code\real_world_data\out" + '\\' + self.name + '.csv'
        isFile = os.path.isfile(path)
        if isFile:
            self.csv_file = path 
            df = pd.read_csv(path)
            df = df.groupby(by='metrics').f1_score.agg(['mean','std']).rename(columns={'mean':'f1_mean', 'std':'f1_stdev'})
            df = df.reset_index().set_index(['metrics'])
            print("\ndf here\n", df)
            self.scores = df.to_dict(orient='index')
    
    def scores_to_table(self, path):
        df = pd.DataFrame(self.scores)
        df = df.round(4)
        df = df.reset_index().drop(['index'], axis='columns')
        for column in df.columns:
            df[column] = df[column].astype('str')
            df.at[0,column] = ', '.join(df[column].astype('str'))
        df = df.drop([1])
        df.insert(0, 'dataset', self.name)
        print("\ndf\n", df)
        df.to_csv(path, index=False, mode='a', header=not os.path.exists(path))


def do_all_unit_tests():
    """ Do all unit tests to reproduce experiments from thesis. """
    target_class = 1
    target_class_name = 'label'

    path_circles = r"Bachelor-Bench\code\unit_tests\data\circles.csv"
    unit_test_circle = Unit_Tests(target_class, target_class_name, 'test_circle', path_circles)
    unit_test_circle.create_new_score()
    unit_test_circle.scores_to_table(unit_test_circle.table_path)

    path_circles_noise = r"Bachelor-Bench\code\unit_tests\data\circles_noise.csv"
    unit_test_noise = Unit_Tests(target_class, target_class_name, 'test_noise', path_circles_noise)
    unit_test_noise.create_new_score()
    unit_test_noise.scores_to_table(unit_test_noise.table_path)

    path_circles_outlier = r"Bachelor-Bench\code\unit_tests\data\circles_outlier.csv"
    unit_test_outlier = Unit_Tests(target_class, target_class_name, 'test_outlier', path_circles_outlier, specific_inst=pd.DataFrame({'x': [-1.3], 'y':[0]}, index=[0]))
    unit_test_outlier.create_new_score()
    unit_test_outlier.scores_to_table(unit_test_outlier.table_path)

    path_circles_imbalance_Bog = r"Bachelor-Bench\code\unit_tests\data\circles_imbalance_Bog.csv"
    unit_test_imbalance_Bog = Unit_Tests(target_class, target_class_name, 'test_imbalance_Bog', path_circles_imbalance_Bog)
    unit_test_imbalance_Bog.create_new_score()
    unit_test_imbalance_Bog.scores_to_table(unit_test_imbalance_Bog.table_path)

    path_circles_imbalance_Bta = r"Bachelor-Bench\code\unit_tests\data\circles_imbalance_Bta.csv"
    unit_test_imbalance_Bag = Unit_Tests(target_class, target_class_name, 'test_imbalance_Bta', path_circles_imbalance_Bta)
    unit_test_imbalance_Bag.create_new_score()
    unit_test_imbalance_Bag.scores_to_table(unit_test_imbalance_Bag.table_path)

    path_blobs = r"Bachelor-Bench\code\unit_tests\data\blobs_many_features.csv"
    unit_test_blobs = Unit_Tests(target_class, target_class_name, 'test_blobs', path_blobs)
    unit_test_blobs.create_new_score()
    unit_test_blobs.scores_to_table(unit_test_blobs.table_path)

    path_circle_close = r"Bachelor-Bench\code\unit_tests\data\circle_close.csv"
    unit_test_circle_close = Unit_Tests(target_class, target_class_name, 'test_circle_close', path_circle_close)
    unit_test_circle_close.create_new_score()
    unit_test_circle_close.scores_to_table(unit_test_circle_close.table_path)


def do_all_eval():
    """ Do all real world tests to reproduce experiments from thesis. """

    target_class = 1

    target_class_name_adm = 'Chance of Admit'
    path_adm = r"Bachelor-Bench\code\real_world_data\data\admission.csv"
    path_adm_model = r"Bachelor-Bench\code\real_world_data\model\model_adm.sav"
    ordering_features_adm = {
        'SOP': ['A', 'B', 'C', 'D', 'F'],
        'LOR': ['A', 'B', 'C', 'D', 'F']
    }
    evaluation_adm = Evaluation(target_class, target_class_name_adm, 'Admission', path_adm, ordering_features_adm, rounds=6, model=path_adm_model)
    evaluation_adm.create_new_score()
    evaluation_adm.scores_to_table(evaluation_adm.table_path)


    target_class_name_credit = 'Risk'
    path_credit = r"Bachelor-Bench\code\real_world_data\data\credit.csv"
    path_credit_model = r"Bachelor-Bench\code\real_world_data\model\model_credit.sav"
    ordering_features_credit = {
        "Job": ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled'],
        "Saving accounts": ['none', 'little', 'moderate', 'quite rich', 'rich'],
        "Checking account": ['none', 'little', 'moderate', 'rich']
    }
    evaluation_credit = Evaluation(target_class, target_class_name_credit, 'Credit', path_credit, ordering_features_credit, svc=False, model=path_credit_model)
    evaluation_credit.create_new_score()
    evaluation_credit.scores_to_table(evaluation_credit.table_path)


    target_class_name_wine = 'quality'
    path_wine = r"Bachelor-Bench\code\real_world_data\data\wine.csv"
    path_wine_model = r"Bachelor-Bench\code\real_world_data\model\model_wine.sav"
    evaluation_wine = Evaluation(target_class, target_class_name_wine, 'Wine', path_wine, svc=True, model=path_wine_model)
    evaluation_wine.create_new_score()
    evaluation_wine.scores_to_table(evaluation_wine.table_path)


    target_class_name_adult = 'income'
    path_adult = r"Bachelor-Bench\code\real_world_data\data\census.csv"
    path_adult_model = r"Bachelor-Bench\code\real_world_data\model\model_adult.sav"
    ordering_features_census_data = {
        "education": ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Some-college', 'Bachelors',  'Masters', 'Doctorate']
    }
    evaluation_census = Evaluation(target_class, target_class_name_adult, 'Census', path_adult, ordering_features_census_data, svc=False, model=path_adult_model)
    evaluation_census.create_new_score()
    evaluation_census.scores_to_table(evaluation_census.table_path)


if __name__ == "__main__":
    
    do_all_unit_tests()

    do_all_eval()