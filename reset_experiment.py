import datetime
import logging
import time

from src.explainer import valid_explainers
from src.io import *
from src.scoring import get_score_df, get_eligible_points_df, get_summary_df
from src.test import valid_tests

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
from contextlib import contextmanager
import threading
import _thread

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()



def get_empty_result(*args):
    return {}
    # return {'score': {},
    #         'time': 0.,
    #         'Last_updated': '',}

TIME_LIMIT = 5 # 250  # src https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call
def run_experiment(test_class, explainer_class):
    print(test_class.__name__, explainer_class.__name__)
    # todo try except
    test = test_class()

    start_time = time.time()

    _explainer = explainer_class(**test.__dict__)
    try:
        with time_limit(TIME_LIMIT, 'sleep'):
            _explainer.explain(dataset_to_explain=test.dataset_to_explain, truth_to_explain=test.truth_to_explain)
    except TimeoutException as e:
        print("Timed out!")
        if _explainer.expected_values is None:
            _explainer.expected_values = f'Time out {TIME_LIMIT}'
        if _explainer.attribution_values is None:
            _explainer.attribution_values =  f'Time out {TIME_LIMIT}'
        if _explainer.feature_importance is None:
            _explainer.feature_importance =  f'Time out {TIME_LIMIT}'

    score = test.score(attribution_values=_explainer.attribution_values,
                       feature_importance=_explainer.feature_importance)
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
    if result_df is None:  # todo [after acceptance] move to io.py
        result_df = pd.DataFrame(index=[e.name for e in valid_explainers],
                                 columns=[t.name for t in valid_tests]).applymap(get_empty_result)
    try:
        for explainer_class in valid_explainers:
            if explainer_class.name not in result_df.index:
                result_df.loc[explainer_class.name] = [get_empty_result() for _ in range(result_df.shape[1])]
            for test_class in valid_tests:
                result = run_experiment(test_class, explainer_class)
                if test_class.name not in result_df.columns:  # todo [after acceptance] check if this line is really important
                    result_df[test_class.name] = [get_empty_result() for _ in range(len(result_df))]
                result_df.at[explainer_class.name, test_class.name] = result
    except KeyboardInterrupt:
        pass
    print(result_df)
    save_results(result_df)

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    summary_df = summary_df.sort_values('time')
    print(summary_df.round(2))

    # todo [after acceptance] record library name and version
