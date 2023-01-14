import itertools
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pareto       # https://github.com/matthewjwoodruff/pareto.py


def invert_column(df, columns):
    """ Invert columns s.t. all should be minimized. """
    for column in columns:
        df[column] = -df[column]
    return df


def get_pareto(dataframe, columns):
    """ Get pareto front of dataframe. """
    objective_columns = dataframe.columns.get_indexer(columns)
    nondominated = pareto.eps_sort([list(dataframe.itertuples(False))], objective_columns)
    df_pareto = pd.DataFrame.from_records(nondominated, columns=list(dataframe.columns.values))
    return df_pareto


def new_order(dataframe, columns, columns_to_invert=[]) -> pd.DataFrame:
    """ Get newly ordered dataframe by pareto optimization. """
    df_loop_constant = dataframe.copy()
    invert_column(df_loop_constant, columns_to_invert)
    newly_ordered = pd.DataFrame()
    while len(df_loop_constant.index) > 0:
        pareto_tmp = get_pareto(df_loop_constant, columns)
        newly_ordered = pd.concat([newly_ordered, pareto_tmp]).reset_index().drop(['index'], axis='columns')
        df_loop_constant = pd.concat([df_loop_constant, pareto_tmp]).drop_duplicates(keep=False)
    invert_column(newly_ordered, columns_to_invert)
    return newly_ordered


def scatter_pareto_combinations_metrics(dataframe, metrics, axs):
    """ Plotting pareto front of all method combinations. """
    combinations = list(itertools.combinations(metrics, 2))
    combinations = [list(a_tuple) for a_tuple in combinations]
    i = 0
    j = 0
    for comb in combinations:
        df_pareto = get_pareto(dataframe, comb)
        sns.scatterplot(data=dataframe, x=comb[0], y=comb[1], ax=axs[i][j], alpha=0.5)
        sns.scatterplot(data=df_pareto, x=comb[0], y=comb[1], ax=axs[i][j], alpha=0.5)
        if j == 0: j = j+1
        else:
            j = 0
            i = i+1
