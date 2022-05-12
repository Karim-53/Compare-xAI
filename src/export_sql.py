import numpy as np
import pandas as pd
import sqlite3 as db

def export_to_sql():
    connection = db.connect('../data/04_sql/database')

    cross_tab = pd.read_parquet('../data/03_experiment_output_aggregated/cross_tab.parquet')
    cross_tab.loc[cross_tab.score.isna(), 'time'] = np.nan  # to not bias the average time per test
    cross_tab.to_sql('cross_tab', connection, if_exists='replace', index=False)

    test = pd.read_csv('../data/01_raw/test.csv')
    test.to_sql('test', connection, if_exists='replace', index=False)

    explainer = pd.read_parquet('../data/03_experiment_output_aggregated/explainer.parquet')
    # explainer.replace({True:1, False:0}, inplace=True)
    explainer = explainer.applymap(lambda x: int(x) if isinstance(x,bool) else x, na_action='ignore')
    explainer.to_sql('explainer', connection, if_exists='replace', index=False)

    paper = pd.read_csv('../data/01_raw/paper.csv')
    paper.to_sql('paper', connection, if_exists='replace', index=False)

    connection.close()
    print('connection.close()')

    import shutil
    shutil.copyfile('../data/04_sql/database', '../../cxai/src/database')

    # verification
    tests = cross_tab.test.unique()
    valid_tests = test.set_index('test').loc[tests]  # if failed: some tests are not indexed in text.csv -> add them
    assert sum(valid_tests['category'].isna()) == 0, 'there is some NaN values -> fill them'
    # todo test that all paper tags in explainer exist in paper


if __name__ == '__main__':
    export_to_sql()
