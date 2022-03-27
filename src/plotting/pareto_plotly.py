from paretoset import paretoset
from plotly import express as px, graph_objects as go

from src.io import load_results
from src.scoring import get_details


def pareto(summary_df, title="Global performance of xAI methods", min_time_value = .01, show=True):
    summary_df.loc[summary_df.time < min_time_value, 'time'] = min_time_value
    if summary_df.percentage.max() < 1.:
        summary_df.percentage = summary_df.percentage * 100
    summary_df['explainer_name'] = summary_df.index

    # see https://plotly.com/python/px-arguments/ for more options
    fig = px.scatter(summary_df,
                     x='time',
                     y='percentage',
                     # s='explainer_name', # https://www.geeksforgeeks.org/how-to-annotate-matplotlib-scatter-plots/
                     # c='eligible_points',
                     text='explainer_name',
                     log_x=True,
                     labels={
                         "time": "Time (seconds)",
                         "percentage": "Percentage %",
                         'eligible_points': 'maximum score',
                         'explainer_name': 'Explainer'
                     },
                     title=title,
                     )

    mask = paretoset(summary_df[['percentage', 'time']], sense=["max", "min"])
    pareto_df = summary_df[mask].sort_values('time')
    fig.add_trace(go.Line(x=pareto_df.time,  #logx=True,
                          y=pareto_df.percentage,
                          text=pareto_df.explainer_name,
                          # textposition='middle right',
                          name='Pareto front',
                          ))
    if show:
        fig.show();
    return fig



from dash import Dash, html, dcc
import pandas as pd

app = Dash(__name__)

result_df = load_results()
summary_df, eligible_points_df, score_df  = get_details(result_df)
fig = pareto(summary_df, show=False)


app.layout = html.Div(children=[
    html.H1(children='Filter:'),

    # html.Div(children='''Dash: A web application framework for your data.'''),

    dcc.Checklist(['I want to explain a specific model']),
    dcc.Dropdown(['model_agnostic', 'tree_based', 'neural_network'],
    disabled=True, multi=True, clearable=True, searchable=True),

    dcc.Checklist(['I need a specific output from the XAI'], ), # todo make the text unselectable # todo chouf el persistance chma3neha
    dcc.Dropdown(options={
        'importance': 'Feature importance (Global Explanation)',
        'attribution': 'Feature attribution (Local Explanation)',
        'interaction': 'Pair feature interaction (Global Explanation)',
        'todo1': 'Todo: Pair interaction (Local Ex), multi F interaction, debugging ...',
    },
    disabled=True, multi=True, clearable=True, searchable=True),

    html.H1(children='Pareto plot:'),
    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
