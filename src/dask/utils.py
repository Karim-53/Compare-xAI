from ast import literal_eval

import pandas as pd

# todo move
try:
    from src.utils import root

    RESULTS_FILE_PATH = root + '/data/02_experiment_output/results.csv'
    deployed = False
except:
    RESULTS_FILE_PATH = root + 'results.csv'
    deployed = True



def load_results(results_file_path=RESULTS_FILE_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            results_file_path,
            index_col=0,
            skipinitialspace=True,
        )
    except FileNotFoundError as e:
        return None  # return pd.DataFrame(index=[e.name for e in valid_explainers], columns=[t.name for t in valid_tests])

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
