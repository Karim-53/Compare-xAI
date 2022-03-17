import logging
import time

# import commentjson
import pandas as pd

from src.explainer import valid_explainers
from src.test import valid_tests

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def run_experiment(test_class, explainer_class):
    print(test_class.__name__, explainer_class.__name__)
    test = test_class()

    start_time = time.time()

    _explainer = explainer_class(**test.__dict__)
    # _explainer = explainer_class(test.trained_model, test.df_train)
    _explainer.explain(test.dataset_to_explain)
    results = test.score(**_explainer.__dict__)

    results['time'] = time.time() - start_time
    return results


def sum_score(dico):
    return sum(dico.values()) - dico['time']


def eligible_points(dico):
    return len(dico) - list(dico.values()).count(None) - 1  # 1 for the time


if __name__ == "__main__":
    result_df = pd.DataFrame(index=valid_explainers.keys(), columns=valid_tests.keys())
    for explainer_name, explainer_class in valid_explainers.items():
        for test_name, test_class in valid_tests.items():
            result = run_experiment(test_class, explainer_class)
            result_df.at[explainer_name, test_name] = result

    score_df = result_df.applymap(sum_score)
    eligible_points_df = result_df.applymap(eligible_points)

    summary_df = pd.DataFrame(index=result_df.index, )
    summary_df['score'] = score_df.sum(axis='columns')
    summary_df['eligible_points'] = eligible_points_df.sum(axis='columns')
    summary_df['percentage'] = summary_df['score'] / summary_df['eligible_points']
    print(summary_df.round(2))
    # todo record library name and version
    # logging.info(predict_func"\nExperiment results : {json.dumps(results, indent=4)}")
    # if not args.no_logs:
    #     parse_utils.save_experiment(experiment, os.path.join(args.results_dir, "checkpoints"), args.rho)
    #     parse_utils.save_results(results, args.results_dir)
    #     parse_utils.save_results_csv(results, args.results_dir)
