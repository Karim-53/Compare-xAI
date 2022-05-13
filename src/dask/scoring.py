import pandas as pd


# todo move out

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
    summary_df['eligible_tests'] = eligible_points_df.applymap(lambda x: 0 if x == 0 else 1).sum(axis='columns')
    summary_df['time_per_test'] = summary_df.time / summary_df.eligible_tests.replace(0, 1)
    return summary_df


def keep_sub_test(k: str, criteria):
    if len(criteria) == 0:
        return True  # no criteria
    if isinstance(criteria, dict):
        assert 'Depricated'
        # for w in ['importance', 'attribution', 'interaction']:
        #     if criteria.get(w, True) and k.startswith(w):
        #         return True
        # return False
    elif isinstance(criteria, list):
        for w in ['importance', 'attribution', 'interaction']:
            if w in criteria and k.startswith(w):
                return True
        return False


def get_details(result_df, verbose=True):
    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    summary_df = summary_df.sort_values('time')
    if verbose:
        print(summary_df.round(2))
    return summary_df, eligible_points_df, score_df


if __name__ == "__main__":
    from src.dask.utils import load_results

    result_df = load_results()
    summary_df, eligible_points_df, score_df = get_details(result_df)


def restrict_tests(
        result_df: pd.DataFrame,
        criteria=None,  # dict or list
        supported_model: str = None,
) -> pd.DataFrame:
    """

    :param result_df:
    :param criteria:
    :param supported_model: None: any (no restriction), 'model-agnostic', 'tree-based'

    :return:
    """
    # just quickly
    if criteria is None and supported_model is None:
        return result_df
    from src.explainer import valid_explainers_dico
    from explainers.explainer_superclass import supported_models_developed

    if supported_model is not None:
        xai_supporting_selected_models = [xai for xai in result_df.index if
                                          supported_model in supported_models_developed(
                                              valid_explainers_dico[xai].supported_models)]
        # print('xai_supporting_selected_models', xai_supporting_selected_models)
    else:
        xai_supporting_selected_models = list(result_df.index)

    if criteria is not None:
        xai_supporting_required_output = [xai for xai in xai_supporting_selected_models
                                          if all(valid_explainers_dico[xai].__dict__.get(c, False) for c in criteria)]
    else:
        xai_supporting_required_output = xai_supporting_selected_models

    if criteria is None:
        criteria = []

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

    _result_df = result_df.loc[xai_supporting_required_output]
    result_df_restricted = _result_df.apply(_restrict_tests, axis=1)
    # pd.DataFrame(columns=result_df.columns, index=result_df.index)

    return result_df_restricted
