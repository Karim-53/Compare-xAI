import datetime
import inspect
import logging
import time

from src.explainer import valid_explainers
from src.io import *
from src.dask.scoring import get_details
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


def empty_results(_iterable):
    return [get_empty_result() for _ in _iterable]


def append_empty_row(result_df, name):
    empty_row_df = pd.DataFrame(
        [empty_results(result_df.columns)],
        index=[name],
        columns=result_df.columns,
    )
    return pd.concat([result_df, empty_row_df])


def compatible(test_class, explainer_class):
    """ test if the xai generate the kind of explanation expected from the test """
    for explanation in ['importance', 'attribution', 'interaction']:
        if explanation in inspect.getfullargspec(test_class.score).args and explainer_class.__dict__.get(explanation,
                                                                                                         False):
            return True
    else:
        return False


TIME_LIMIT = 1000  # 250  # src https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call


def run_experiment(test_class, explainer_class):
    print(test_class.__name__, explainer_class.__name__)
    if not compatible(test_class, explainer_class):
        return {
            # 'score': score,
            'Last_updated': str(datetime.datetime.now()),
        }
    # todo try except to catch error from XAI alg
    test = test_class()

    start_time = time.time()

    # print(test.__dict__)
    _explainer = explainer_class(**test.__dict__)
    try:
        with time_limit(TIME_LIMIT, 'sleep'):
            _explainer.explain(dataset_to_explain=test.dataset_to_explain, truth_to_explain=test.truth_to_explain)
    except TimeoutException as e:
        print("Timed out!")
        if _explainer.__dict__.get('expected_values', None) is None:
            _explainer.expected_values = f'Time out {TIME_LIMIT}'
        if _explainer.__dict__.get('attribution', None) is None:
            _explainer.attribution = f'Time out {TIME_LIMIT}'
        if _explainer.__dict__.get('importance', None) is None:
            _explainer.importance = f'Time out {TIME_LIMIT}'

    score = test.score(attribution=_explainer.attribution,
                       importance=_explainer.importance,
                       interaction=_explainer.interaction,
                       )
    results = {
        'score': score,
        'time': time.time() - start_time,
        'Last_updated': str(datetime.datetime.now()),
    }
    return results


if __name__ == "__main__":
    print(f'Explainers: {len(valid_explainers)} / 12 pour le 04-08')
    print(f'Tests: {len(valid_tests)} / 24 pour le 04-08')
    result_df = load_results()
    if result_df is None:  # todo [after acceptance] move to io.py
        result_df = pd.DataFrame(index=[e.name for e in valid_explainers],
                                 columns=[t.name for t in valid_tests]).applymap(get_empty_result)
    try:
        for explainer_class in valid_explainers:
            if explainer_class.name not in result_df.index:
                result_df = append_empty_row(result_df, explainer_class.name)
            for test_class in valid_tests:
                if test_class.name not in result_df.columns:  # todo [after acceptance] check if this line is really important
                    result_df[test_class.name] = empty_results(result_df.index)
                result = run_experiment(test_class, explainer_class)
                print('old result', result_df.loc[explainer_class.name, test_class.name])
                result_df.at[explainer_class.name, test_class.name] = result
                print('new result', result_df.loc[explainer_class.name, test_class.name])
    except KeyboardInterrupt:
        pass
    print(result_df)
    save_results_safe(result_df)

    summary_df, eligible_points_df, score_df = get_details(result_df)

    # todo [after acceptance] record library name and version
