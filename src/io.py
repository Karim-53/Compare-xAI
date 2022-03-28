import os
from ast import literal_eval

import pandas as pd

from src.test import get_sub_tests
from src.utils import root

RESULTS_FILE_PATH = root + '/results/results.csv'
RESULTS_TMP_FILE_PATH = root + '/results/results_tmp.csv'


def load_results(results_file_path=RESULTS_FILE_PATH) -> pd.DataFrame:
    if not os.path.exists(results_file_path):
        return None  # return pd.DataFrame(index=[e.name for e in valid_explainers], columns=[t.name for t in valid_tests])

    df = pd.read_csv(
        results_file_path,
        index_col=0,
        skipinitialspace=True,
    )
    try:
        df = df.applymap(literal_eval)
    except:
        for idx in df.index:
            for col in df.columns:
                try:
                    e = None
                    e = df.loc[idx, col]
                    literal_eval(e)
                except:
                    print(f'Failed to eval df[{idx}, {col}] = ', e, 'of type', type(e))
        raise ValueError('Unable to load results_df')

    df.columns = [c.replace(' ', '') for c in df.columns]
    df.index = [i.replace(' ', '') for i in df.index]
    return df


def save_results(result_df: pd.DataFrame):
    # print('writing to', results_file_path, '...')
    result_df.to_csv(RESULTS_FILE_PATH)


def save_results_safe(result_df: pd.DataFrame):
    print('writing to', RESULTS_FILE_PATH, '...')
    try:
        result_df.to_csv(RESULTS_TMP_FILE_PATH)
        if load_results(RESULTS_TMP_FILE_PATH) is not None:
            save_results(result_df)
    except:
        print('Failed to save the results correctly:')
        raise
    # todo [after acceptance] write summary_df, eligible_points_df, score_df to keep track of the progress


def insert_meta_row(sub_test, test_name, row, lista):
    d = {'test': test_name,
         'sub_test_category': 'meta',
         'sub_test': sub_test,
         }
    for i, val in row.items():
        d[i] = val.get(sub_test)
    lista.append(d)

def insert_score_row(sub_test, test_name, row, lista):
    d = {'test': test_name,
         'sub_test_category': 'score',
         'sub_test': sub_test,
         }
    for i, val in row.items():
        d[i] = val.get('score', {}).get(sub_test, None)
    lista.append(d)


def detail(result_df): # todo move to scoring
    """ return a multi index df ['test', 'sub_test_category', 'sub_test'] vs explainer in columns"""
    lista = []
    for test_name, row in result_df.T.iterrows():

        insert_meta_row('time', test_name, row, lista)
        insert_meta_row('Last_updated', test_name, row, lista)
        sub_tests = get_sub_tests(test_name)
        for sub_test in sub_tests:
            insert_score_row(sub_test, test_name, row, lista)
    return pd.DataFrame(lista)  # .set_index(['test', 'sub_test_category', 'sub_test'])


if __name__ == "__main__":
    from src.scoring import get_score_df, get_eligible_points_df, get_summary_df

    result_df = load_results()
    print(result_df)
    print('cell type:', type(result_df.iloc[0, 0]))

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))
