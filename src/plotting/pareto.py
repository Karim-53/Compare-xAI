# if __name__ == "__main__":
#     """ Pareto with matplotlib (depricated) """
#     from src.scoring import get_score_df, get_eligible_points_df, get_summary_df
#
#     result_df = load_results()
#     print(result_df)
#     print('cell type:', type(result_df.iloc[0, 0]))
#
#     score_df = get_score_df(result_df)
#     eligible_points_df = get_eligible_points_df(result_df)
#     summary_df = get_summary_df(result_df, score_df, eligible_points_df)
#     print(summary_df.round(2))
#     min_x_axis_value = .01
#     summary_df.loc[summary_df.time < min_x_axis_value, 'time'] = min_x_axis_value
#     summary_df['explainer_name'] = summary_df.index
#     ax = summary_df.plot.scatter(x='time',
#                                  y='percentage',
#                                  # s='explainer_name', # https://www.geeksforgeeks.org/how-to-annotate-matplotlib-scatter-plots/
#                                  c='eligible_points',
#                                  )
#     # must be an interactive plot
#     mask = paretoset(summary_df[['percentage', 'time']], sense=["max", "min"])
#     pareto_df = summary_df[mask].sort_values('time')
#     pareto_df.plot.line(x='time', logx=True,
#                         y='percentage',
#                         ax=ax,
#                         )
#     plt.legend(['Pareto'], loc='lower right')
#     plt.ylim([0, 1])
#     plt.show()
