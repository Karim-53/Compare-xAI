from paretoset import paretoset
from plotly import express as px, graph_objects as go

from src.io import load_results
from src.scoring import get_details, restrict_tests


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

app = Dash(__name__)

result_df = load_results()
summary_df, eligible_points_df, score_df  = get_details(result_df, verbose=False)
fig = pareto(summary_df, show=False)


app.layout = html.Div(children=[
    html.H1(children='Filter:'),

    # html.Div(children='''Dash: A web application framework for your data.'''),

    dcc.Checklist(id='required_outputs_checklist',
                  options={'specific_xai_output': 'I want to explain a specific model'}),  # todo [after acceptance] Learn more (link)
    dcc.Dropdown(id='required_outputs_dropdown',
                 options=['model_agnostic', 'tree_based', 'neural_network'],
                 value=['model_agnostic'],
                 disabled=True, clearable=True, searchable=True,
                 # multi=True,  # todo
                 ),

    # dcc.Checklist(id='supported_models_checklist',
    #               options=['I need a specific output from the XAI'],  # todo [after acceptance] Learn more (link)
    #               ),  # todo make the text unselectable # todo chouf el persistance chma3neha
    # dcc.Dropdown(id='supported_models_dropdown',
    #              options={
    #                  'importance': 'Feature importance (Global Explanation)',
    #                  'attribution': 'Feature attribution (Local Explanation)',
    #                  'interaction': 'Pair feature interaction (Global Explanation)',
    #                  'todo1': 'Todo: Pair interaction (Local Ex), multi F interaction, debugging ...',
    #              },
    #              disabled=True, multi=True, clearable=True, searchable=True),

    html.H1(children='Pareto plot:'),
    dcc.Graph(
        id='pareto_plot',
        figure=fig
    )
])
from dash import Input, Output, State


@app.callback(
    Output(component_id='required_outputs_dropdown', component_property='disabled'),
    Output(component_id='pareto_plot', component_property='figure'),

    Input(component_id='required_outputs_checklist', component_property='value'),
    Input(component_id='required_outputs_dropdown', component_property='value'),
    # Input(component_id='supported_models_checklist', component_property='value'),
    # Input(component_id='supported_models_dropdown', component_property='value'),
)
def update_output_div(required_outputs_checklist,
                      required_outputs_dropdown,
                      # supported_models_checklist,
                      # supported_models_dropdown,
                      ):
    supported_model = None

    if required_outputs_checklist is not None and 'specific_xai_output' in required_outputs_checklist:
        required_outputs_dropdown_disabled = False
        if required_outputs_dropdown is not None:
            supported_model = required_outputs_dropdown
    else:
        required_outputs_dropdown_disabled = True

    print(
      required_outputs_checklist,
      required_outputs_dropdown,
      # supported_models_checklist,
      # supported_models_dropdown,
          )

    result_df_restricted_tree = restrict_tests(result_df,
                                               criteria={},
                                               supported_model=supported_model)
    summary_df_restricted_tree, _, _  = get_details(result_df_restricted_tree, verbose=True)
    # print(summary_df_restricted_tree.columns)
    # summary_df_restricted_tree = summary_df
    fig = pareto(summary_df_restricted_tree, show=False)
    return (required_outputs_dropdown_disabled, fig)
    # return f'Output: {input_value}'


if __name__ == '__main__':
    app.run_server(debug=True)
