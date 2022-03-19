import matplotlib.pyplot as plt

from src.io import load_results

if __name__ == "__main__":
    from src.scoring import get_score_df, get_eligible_points_df, get_summary_df

    result_df = load_results()
    print(result_df)
    print('cell type:', type(result_df.iloc[0, 0]))

    score_df = get_score_df(result_df)
    eligible_points_df = get_eligible_points_df(result_df)
    summary_df = get_summary_df(result_df, score_df, eligible_points_df)
    print(summary_df.round(2))
    ax1 = summary_df.plot.scatter(x='time',
                                  y='percentage',
                                  c='eligible_points')
    # must be an interactive plot
    plt.show()
