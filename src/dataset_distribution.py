""" plot the distribution of score and time"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.dask.utils import load_results
from src.dask.scoring import get_score_df, get_eligible_points_df, get_summary_df

if __name__ == "__main__":

    result_df = load_results()
    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    # print(summary_df.round(2))


    lista = []
    for idx, row in result_df.iterrows():
        for dico in row:
            for v in dico.get('score', {}).values():
                if not pd.isna(v):
                    lista.append(v)
    # distrib of scores (real)
    fig = sns.histplot(lista, kde=False, binwidth=0.05, label='real data', legend=True)
    # sns.histplot(np.random.beta(.6, .6, 2*len(lista)), kde=True, label='approximation', color='black', ax=plt.gca())  # distrib of scores (approx.)
    # it is not a good approx because kernel can only be gaussian :(
    # plt.legend()
    plt.show()

    # distrib of average scores (real)
    sns.histplot(summary_df.percentage, kde=False, label='real avg. scores', legend=True)
    # too few values :/
    plt.show()

    lista_time = []
    for idx, row in result_df.iterrows():
        for dico in row:
            v = dico.get('time')
            if not pd.isna(v):
                lista_time.append(v)
    # distrib of time (real)
    fig = sns.histplot(lista_time, kde=True, label='real time', legend=True)
    plt.xlim([None,200])
    plt.show()
