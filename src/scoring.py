from typing import Dict

import pandas as pd



def sum_score(dico):
    score = dico.get('score', {})
    score_not_none = [f for f in score.values() if isinstance(f, float)]
    return sum(score_not_none)


def get_score_df(result_df):
    return result_df.applymap(sum_score)


def eligible_points(dico):
    score = dico.get('score', {})
    score_not_none = [f for f in score.values() if isinstance(f, float)]
    return len(score_not_none)


def get_eligible_points_df(result_df):
    return result_df.applymap(eligible_points)


def get_summary_df(result_df: pd.DataFrame, score_df: pd.DataFrame, eligible_points_df: pd.DataFrame) -> pd.DataFrame:
    # todo [after acceptance] eligible_points_df could be = None and infered from result_df
    summary_df = pd.DataFrame(index=result_df.index, )
    summary_df['time'] = result_df.apply(
        lambda series: sum([dico.get('time', 0.) for test_name, dico in series.items()]),
        axis='columns')
    summary_df['score'] = score_df.sum(axis='columns')
    summary_df['eligible_points'] = eligible_points_df.sum(axis='columns')
    summary_df['percentage'] = summary_df['score'] / summary_df['eligible_points']
    return summary_df


def get_details(result_df):
    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))
    return summary_df, eligible_points_df, score_df


def keep_sub_test(k: str, criteria):
    for w in ['importance', 'attribution', 'interaction']:
        if criteria.get(w, True) and k.startswith(w):
            return True
    return False


def restrict_tests(
        result_df: pd.DataFrame,
        criteria: Dict[str, bool] = {},
        supported_model: str = None,
) -> pd.DataFrame:
    """

    :param result_df:
    :param criteria:
    :param supported_model: None: any (no restriction), 'model-agnostic', 'tree-based'

    :return:
    """
    from src.explainer import valid_explainers_dico
    from explainers.explainer_superclass import supported_models_developed

    if supported_model is not None:
        xai_supporting_slected_models = [xai for xai in result_df.index if supported_model in supported_models_developed(valid_explainers_dico[xai].supported_models) ]
    else:
        xai_supporting_slected_models = result_df.index
    def _restrict_tests(row):
        # if supported_model is not None:
        #     if supported_model not in supported_models_developed(valid_explainers_dico[row.name].supported_models):
        #         return None  # no need to keep the column as the model is not considered
        restricted_results = []
        for dico in row:
            sub_tests = dico.get('score', {})
            new_score = {}
            for k, v in sub_tests.items():
                if keep_sub_test(k, criteria):
                    new_score[k] = v
            if len(new_score):
                new_result = {
                    'score': new_score,
                    'time': dico.get('time', 0.)
                }
            else:
                new_result = {}
            restricted_results.append(new_result)
        return pd.Series(restricted_results, name=row.name, index=result_df.columns)

    _result_df = result_df.loc[xai_supporting_slected_models]
    result_df_restricted = _result_df.apply(_restrict_tests, axis=1)
    # pd.DataFrame(columns=result_df.columns, index=result_df.index)

    return result_df_restricted


if __name__ == "__main__":
    from src.io import load_results

    result_df = load_results()
    summary_df, eligible_points_df, score_df  = get_details(result_df)

    print('Best XAI for feature importance')
    result_df_restricted = restrict_tests(result_df,
                                          criteria={'importance': True,
                                                    'attribution': False,
                                                    'interaction': False, },
                                          supported_model=None)
    summary_df_restricted, eligible_points_df_restricted, score_df_restricted  = get_details(result_df_restricted)

    print('Best XAI for explaining feature importance of tree models')
    result_df_restricted = restrict_tests(result_df,
                                          criteria={'importance': True,
                                                    'attribution': False,
                                                    'interaction': False, },
                                          supported_model='tree_based')
    summary_df_restricted, eligible_points_df_restricted, score_df_restricted  = get_details(result_df_restricted)

    print('Best model-agnostic XAI')
    result_df_restricted = restrict_tests(result_df,
                                          criteria={},
                                          supported_model='model_agnostic')
    summary_df_restricted, eligible_points_df_restricted, score_df_restricted  = get_details(result_df_restricted)
