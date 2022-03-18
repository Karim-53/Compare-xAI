import logging
import time

import pandas as pd

from src.explainer import valid_explainers
from src.io import load_results, save_results
from src.scoring import get_score_df, get_eligible_points_df, get_summary_df
from src.test import valid_tests

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_experiment(test_class, explainer_class):
    print(test_class.__name__, explainer_class.__name__)
    # todo try except
    test = test_class()

    start_time = time.time()

    _explainer = explainer_class(**test.__dict__)
    # _explainer = explainer_class(test.predict_func, test.df_train)
    _explainer.explain(test.dataset_to_explain)
    results = test.score(**_explainer.__dict__)

    results['time'] = time.time() - start_time
    return results


if __name__ == "__main__":
    result_df = load_results()
    if result_df is None:   # todo [after acceptance] move to io.py
        result_df = pd.DataFrame(index=valid_explainers.keys(), columns=valid_tests.keys())

    for explainer_name, explainer_class in valid_explainers.items():
        for test_name, test_class in valid_tests.items():
            result = run_experiment(test_class, explainer_class)
            result_df.at[explainer_name, test_name] = result
    save_results(result_df)

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))


    # todo [after acceptance] record library name and version
    # todo record results as csv
    # logging.info(predict_func"\nExperiment results : {json.dumps(results, indent=4)}")
    # if not args.no_logs:
    #     parse_utils.save_experiment(experiment, os.path.join(args.results_dir, "checkpoints"), args.rho)
    #     parse_utils.save_results(results, args.results_dir)
    #     parse_utils.save_results_csv(results, args.results_dir)
