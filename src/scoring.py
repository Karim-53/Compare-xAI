import pandas as pd


def sum_score(dico):
    score = dico.get('score', {})
    return sum(score.values())


def get_score_df(result_df):
    return result_df.applymap(sum_score)


def eligible_points(dico):
    score = dico.get('score', {})
    return sum([isinstance(s, float) for s in score.values()])


def get_eligible_points_df(result_df):
    return result_df.applymap(eligible_points)


def get_summary_df(result_df: pd.DataFrame, score_df: pd.DataFrame, eligible_points_df: pd.DataFrame) -> pd.DataFrame:
    # todo [after acceptance] eligible_points_df could be = None and infered from result_df
    summary_df = pd.DataFrame(index=result_df.index, )
    summary_df['time'] = result_df.apply(lambda series: sum([dico.get('time', 0.) for test_name, dico in series.items()]),
                                         axis='columns')
    summary_df['score'] = score_df.sum(axis='columns')
    summary_df['eligible_points'] = eligible_points_df.sum(axis='columns')
    summary_df['percentage'] = summary_df['score'] / summary_df['eligible_points']
    return summary_df