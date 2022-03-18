import os
from ast import literal_eval

import pandas as pd

from src.utils import root

results_file_path = root + '/results/results.csv'


def load_results() -> pd.DataFrame:
    if os.path.exists(results_file_path):
        return pd.read_csv(results_file_path, index_col=0).applymap(literal_eval)
    else:
        return None  # return pd.DataFrame(index=valid_explainers.keys(), columns=valid_tests.keys())


def save_results(result_df: pd.DataFrame):
    print('writing to', results_file_path, '...')
    result_df.to_csv(results_file_path)


if __name__ == "__main__":
    from src.scoring import get_score_df, get_eligible_points_df, get_summary_df

    result_df = load_results()
    print(result_df)
    print('cell type:', type(result_df.iloc[0, 0]))

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))