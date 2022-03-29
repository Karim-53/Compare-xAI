from paretoset import paretoset
from plotly import express as px, graph_objects as go

from src.io import load_results
from src.scoring import get_details, restrict_tests

# todo [after acceptance] peu etre nzid: dot size eligible points
def pareto(summary_df, title="Global performance of xAI methods", min_time_value = .01, show=True):
    assert len(summary_df)>0, 'No XAI to plot. At least the baseline_random should be there'
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
                         "time": "Time (seconds) ",
                         "percentage": "Percentage % ",
                         'eligible_points': 'maximum score ',
                         'explainer_name': 'Explainer '
                     },
                     title=title,
                     )

    mask = paretoset(summary_df[['percentage', 'time']], sense=["max", "min"])
    pareto_df = summary_df[mask].sort_values('time')
    txt = "Time (seconds) = " + pareto_df.time.round(0).astype(str) + '\nPercentage % = ' + pareto_df.percentage.round(0).astype(str) + '\neligible_points = ' + pareto_df.eligible_points.astype(str) + '\n\nExplainer = ' + pareto_df.explainer_name
    fig.add_trace(go.Line(x=pareto_df.time,  #logx=True,
                          y=pareto_df.percentage,
                          text=txt,
                          # textposition='middle right',
                          name='Pareto front',
                          ))
    fig.update_traces(textposition='top center')

    if show:
        fig.show();
    return fig

def get_stats(eligible_points_df):
    xai = len(eligible_points_df)
    tests = eligible_points_df.max().sum()
    return xai,tests


from dash import Dash, html, dcc

app = Dash(__name__, prevent_initial_callbacks=True)

result_df = load_results()
summary_df, eligible_points_df, score_df = get_details(result_df, verbose=True)
max_xai, max_tests = get_stats(eligible_points_df)

def get_text_stats(eligible_points_df):
    xai,tests = get_stats(eligible_points_df)
    return [f"Kept XAI {xai} / {max_xai}", html.Br(), f"Kept tests {tests} / {max_tests}"]


text_stats = get_text_stats(eligible_points_df)
fig = pareto(summary_df, show=False)

todo_lista = ['I trust the XAI output (I created the data and the model myself)',
'I know the target value of the data points to explain',
'I can retrain my model',
'I can perform additional predictions',
'I have a reference input data',
'I have a GPU ',]

# todo [after acceptance] on hover help / tips
# todo button reset
# todo Click on the dot will take you to the git page with the csv of that method
app.layout = html.Div(children=[
    html.H1(children='Filter:'),

    dcc.Checklist(id='supported_models_checklist',
                  options={'specific_supported_model': 'I want to explain a specific model'}),
    # todo [after acceptance] Learn more (link)
    dcc.Dropdown(id='supported_models_dropdown',
                 options=['model_agnostic', 'tree_based', 'neural_network'],
                 value=['model_agnostic'],
                 disabled=True, clearable=True, searchable=True,
                 # multi=True,  # todo
                 ),

    dcc.Checklist(id='required_outputs_checklist',
                  options={'specific_xai_output': 'I need a specific output from the XAI',},
                  ),  # todo make the text unselectable # todo chouf el persistance chma3neha
    dcc.Dropdown(id='required_outputs_dropdown',
                 options={
                     'importance': 'Feature importance (Global Explanation)',
                     'attribution': 'Feature attribution (Local Explanation)',
                     'interaction': 'Pair feature interaction (Global Explanation)',
                     'todo1': 'Todo: Pair interaction (Local Ex), multi F interaction, debugging ...',
                 },
                 disabled=True, multi=True, clearable=True, searchable=True),

    dcc.Checklist(id='todo',
                  options=todo_lista,
                  value=todo_lista,
                  ),
    html.Div(html.P(id='kept_objects', children=get_text_stats(eligible_points_df))),

    html.H1(children='Pareto plot:'),
    dcc.Graph(
        id='pareto_plot',
        figure=fig
    )
])
from dash import Input, Output


@app.callback(
    Output(component_id='supported_models_dropdown', component_property='disabled'),
    Output(component_id='required_outputs_dropdown', component_property='disabled'),
    Output(component_id='pareto_plot', component_property='figure'),
    Output(component_id='kept_objects', component_property='children'),

    Input(component_id='supported_models_checklist', component_property='value'),
    Input(component_id='supported_models_dropdown', component_property='value'),
    Input(component_id='required_outputs_checklist', component_property='value'),
    Input(component_id='required_outputs_dropdown', component_property='value'),
)
def update_output_div(
        supported_models_checklist,
        supported_models_dropdown,
        required_outputs_checklist,
        required_outputs_dropdown,
):
    print(
        supported_models_checklist,
        supported_models_dropdown,
        required_outputs_checklist,
        required_outputs_dropdown,
    )

    supported_model = None
    required_outputs = None
    supported_models_dropdown_disabled = True
    required_outputs_dropdown_disabled = True

    if supported_models_checklist is not None and 'specific_supported_model' in supported_models_checklist:
        supported_models_dropdown_disabled = False
        if supported_models_dropdown is not None:
            supported_model = supported_models_dropdown
            if isinstance(supported_model,list):
                supported_model = supported_model[0]
    else:
        supported_models_dropdown_disabled = True

    if required_outputs_checklist is not None and 'specific_xai_output' in required_outputs_checklist:
        required_outputs_dropdown_disabled = False
        if required_outputs_dropdown is not None:
            required_outputs = required_outputs_dropdown
    else:
        required_outputs_dropdown_disabled = True

    result_df_restricted = restrict_tests(result_df,
                                          criteria=required_outputs,
                                          supported_model=supported_model)
    print(result_df_restricted.index)
    summary_df_restricted_tree, eligible_points_df_restricted, _ = get_details(result_df_restricted, verbose=True)
    p_txt = get_text_stats(eligible_points_df_restricted)
    # print(p_txt)
    fig = pareto(summary_df_restricted_tree, show=False)
    return supported_models_dropdown_disabled, required_outputs_dropdown_disabled, fig, p_txt


if __name__ == '__main__':
    app.run_server(debug=False, port=8005)  #  dev_tools_hot_reload=False,
