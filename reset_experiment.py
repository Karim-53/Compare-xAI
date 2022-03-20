import datetime
import logging
import time

import pandas as pd

from src.explainer import valid_explainers
from src.io import *
from src.scoring import get_score_df, get_eligible_points_df, get_summary_df
from src.test import valid_tests

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_empty_result(*args):
    return {}
    # return {'score': {},
    #         'time': 0.,
    #         'Last_updated': '',}


def run_experiment(test_class, explainer_class):
    print(test_class.__name__, explainer_class.__name__)
    # todo try except
    test = test_class()

    start_time = time.time()

    _explainer = explainer_class(**test.__dict__)
    # _explainer = explainer_class(test.predict_func, test.df_train)
    _explainer.explain(dataset_to_explain=test.dataset_to_explain, truth_to_explain=test.truth_to_explain)
    # print('----', _explainer.attribution_values)
    score = test.score(attribution_values=_explainer.attribution_values, feature_importance=_explainer.feature_importance)
    results = {
        'score': score,
        'time': time.time() - start_time,
        'Last_updated': str(datetime.datetime.now()),
    }
    return results


if __name__ == "__main__":
    print(f'Explainers: {len(valid_explainers)}')
    print(f'Tests: {len(valid_tests)}')
    result_df = load_results()
    if result_df is None:   # todo [after acceptance] move to io.py
        result_df = pd.DataFrame(index=[e.name for e in valid_explainers], columns=[t.name for t in valid_tests]).applymap(get_empty_result)

    for explainer_class in valid_explainers:
        if explainer_class.name not in result_df.index:
            result_df.loc[explainer_class.name] = [get_empty_result() for _ in range(result_df.shape[1])]
        for test_class in valid_tests:
            result = run_experiment(test_class, explainer_class)
            if test_class.name not in result_df.columns:   # todo [after acceptance] check if this line is really important
                result_df[test_class.name] = [get_empty_result() for _ in range(len(result_df))]
            result_df.at[explainer_class.name, test_class.name] = result
    print(result_df)
    save_results(result_df)

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))

    # todo [after acceptance] record library name and version