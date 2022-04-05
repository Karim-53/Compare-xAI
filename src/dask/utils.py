import os
from ast import literal_eval

import pandas as pd

try:
    from src.utils import root
    RESULTS_FILE_PATH = root + '/src/dask/results.csv'
    deployed = False
except:
    RESULTS_FILE_PATH = root + 'results.csv'
    deployed = True

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


RESULTS_TMP_FILE_PATH = root + '/src/dask/results_tmp.csv'