""" Main script to run the Pareto plot online
check the remaining hours https://dashboard.heroku.com/account/billing
"""
# todo delete
# TODO stop using backend, use plotly.offline https://stackoverflow.com/questions/46821554/multiple-plotly-plots-on-1-page-without-subplot/59265030#59265030
import visdcc

try:
    from pprint import pprint
except:
    pprint = print

import pandas as pd
from paretoset import paretoset
from plotly import express as px, graph_objects as go

from utils import load_results
from scoring import get_details, restrict_tests
from dash import Dash, html, dcc, Input, Output

# todo [after acceptance] dot size legend
# from src.utils import root  # local website
root = 'https://karim-53.github.io/Compare-xAI/'
explainer_root_link = root + 'explainers/'


def pareto(summary_df, min_time_value=.01, show=True):
    """ Generate the scatter plot and the pareto front given the specific selection of xai and unit-tests """
    assert len(summary_df) > 0, 'No XAI to plot. At least the baseline_random should be there'
    summary_df.loc[summary_df.time_per_test < min_time_value, 'time_per_test'] = min_time_value
    if summary_df.percentage.max() < 1.:
        summary_df.percentage = summary_df.percentage * 100
    summary_df['explainer_name'] = summary_df.index
    # see https://plotly.com/python/px-arguments/ for more options
    # title = ""
    fig = px.scatter(summary_df,
                     x='time_per_test',
                     y='percentage',
                     # here I should not show score because 2 xai can have same score 3/3 and 3/10 and it is misleading
                     # s='explainer_name', # https://www.geeksforgeeks.org/how-to-annotate-matplotlib-scatter-plots/
                     # c='eligible_points',
                     size='eligible_points',
                     text='explainer_name',
                     log_x=True,
                     labels={
                         "time_per_test": "Average Time per test [Seconds] ↓",
                         "percentage": "Score [%] ↑",
                         'eligible_points': 'maximum score ',
                         'explainer_name': 'Explainer '
                     },
                     # title=title,  # take some vertical space
                     )

    # Pareto front -------------------------------------------------------------------
    mask = paretoset(summary_df[['percentage', 'time_per_test']], sense=["max", "min"])
    pareto_df = summary_df[mask].sort_values('time_per_test')
    txt = "Avg. Time = " + pareto_df.time_per_test.round(1).astype(str)
    txt += ' <br>Score = ' + pareto_df.percentage.round(1).astype(str) + ' %'
    txt += ' <br>eligible_points = ' + pareto_df.eligible_points.astype(str)
    txt += ' <br>Explainer = ' + pareto_df.explainer_name
    fig.add_trace(go.Line(x=pareto_df.time_per_test,  # logx=True,
                          y=pareto_df.percentage,
                          text=txt,
                          # textposition='middle right',
                          name='Pareto front',
                          ))
    fig.update_traces(textposition='top center')

    if show:
        fig.show()
    return fig


def get_stats(eligible_points_df):
    """ return the number of xai and unit-tests selected """
    xai = len(eligible_points_df)
    tests = eligible_points_df.max().sum()
    return xai, tests


app = Dash(__name__, prevent_initial_callbacks=True)
server = app.server

result_df = load_results()
summary_df, eligible_points_df, score_df = get_details(result_df, verbose=True)
max_xai, max_tests = get_stats(eligible_points_df)


def get_text_stats(eligible_points_df):
    xai, tests = get_stats(eligible_points_df)
    return [f"Kept XAI {xai} / {max_xai}", html.Br(), f"Kept tests {tests} / {max_tests}"]


text_stats = get_text_stats(eligible_points_df)
fig = pareto(summary_df, show=False)

todo_lista = ['I trust the XAI output (I created the data and the model myself)',
              # requirements from the xai alg: requirements on the data
              'I have a reference input data',
              'I know the target value of the data points to explain',  # both global and local explanation
              'Assume feature distribution iid',
              # requirement from the xai alg: 
              'The XAI can retrain my model',
              'The XAI can perform additional predictions',
              'I have a GPU ', ]

# todo [after acceptance] on hover help / tips
# todo button reset
app.layout = html.Div(children=[
    html.H1(children='Filter:'),

    dcc.Checklist(id='supported_models_checklist',
                  options={'specific_supported_model': 'I want to explain a specific model'}),
    # todo [after acceptance] Learn more (link)
    dcc.Dropdown(id='supported_models_dropdown',
                 options=['model_agnostic', 'tree_based', 'neural_network'],
                 value=['model_agnostic'],
                 style={'display': 'none'}, clearable=True, searchable=True,
                 # multi=True,  # todo
                 ),

    dcc.Checklist(id='required_outputs_checklist',
                  options={'specific_xai_output': 'I need a specific output from the XAI', },
                  ),
    dcc.Dropdown(id='required_outputs_dropdown',
                 options={
                     'importance': 'Feature importance (Global Explanation)',
                     'attribution': 'Feature attribution (Local Explanation)',
                     # We discuss the attribution problem, i.e., the problem of distributing the prediction score of a model for a specific input to its base features (cf. [15, 10, 19]); the attribution to a base feature can be interpreted as the importance of the feature to the prediction. https://arxiv.org/pdf/1908.08474.pdf
                     'interaction': 'Pair feature interaction (Global Explanation)',
                     # Definition 1 (Statistical Non-Additive Interaction). A function f contains a statistical non-additive interaction of multiple features indexed in set I if and only if f cannot be decomposed into a sum of |I| subfunctions fi , each excluding the i-th interaction variable: f(x) =/= Sum i∈I fi(x\{i}).
                     #  Def. 1 identifies a non-additive effect among all features I on the output of function f (Friedman and Popescu, 2008; Sorokina et al., 2008; Tsang et al., 2018a). see https://arxiv.org/pdf/2103.03103.pdf
                     # todo [after acceptance] we need a page with a clear description of each option
                     'todo1': 'Todo: Pair interaction (Local Ex), multi F interaction, NLP, debugging ...',
                 },
                 style={'display': 'none'}, multi=True, clearable=True, searchable=True),

    html.P(id='help_txt2',
           children='The Checklist below do not affect the plot yet (just for demonstration purpose):'),

    dcc.Checklist(id='todo',
                  options=todo_lista,
                  value=todo_lista,
                  inline=False,
                  ),
    html.P(id='kept_objects', children=get_text_stats(eligible_points_df)),

    html.H1(children='Pareto plot: Global performance of xAI methods'),
    html.P(id='help_txt',
           children='Dot size == how much polyvalent is the xAI (i.e. this xAI works on different explanation tasks). higher is better. ______________ Click on a dot to know more about the xAI'),
    dcc.Graph(
        id='pareto_plot',
        figure=fig
    ),
    visdcc.Run_js(id='javascript'),
])

last_click_data = ''


def on_click(clickData) -> str:
    """ return Bool if the event is correct """
    if pd.isna(clickData):
        return ''
    global last_click_data
    if last_click_data == str(clickData):
        return ''
    last_click_data = str(clickData)
    pprint(clickData)
    txt = clickData.get('points', [{}])[0].get('text', '')
    if txt == '':
        return ''
    sub_str = '<br>Explainer = '
    pos = txt.find(sub_str)
    explainer_name = txt
    if pos != -1:
        explainer_name = txt[pos + len(sub_str):]
    print('explainer_name', explainer_name)
    url = explainer_root_link + explainer_name + '.htm'
    print('opening:', url)
    return f"window.open('{url}')"


@app.callback(
    Output(component_id='supported_models_dropdown', component_property='style'),
    Output(component_id='required_outputs_dropdown', component_property='style'),
    Output(component_id='pareto_plot', component_property='figure'),
    Output(component_id='kept_objects', component_property='children'),
    Output('javascript', 'run'),

    Input('pareto_plot', 'clickData'),
    Input(component_id='supported_models_checklist', component_property='value'),
    Input(component_id='supported_models_dropdown', component_property='value'),
    Input(component_id='required_outputs_checklist', component_property='value'),
    Input(component_id='required_outputs_dropdown', component_property='value'),
)
def update_output_div(
        clickData,
        supported_models_checklist,
        supported_models_dropdown,
        required_outputs_checklist,
        required_outputs_dropdown,
):
    # print(
    #     supported_models_checklist,
    #     supported_models_dropdown,
    #     required_outputs_checklist,
    #     required_outputs_dropdown,
    # )
    url = on_click(clickData)

    supported_model = None
    required_outputs = None
    supported_models_dropdown_disabled = True
    required_outputs_dropdown_disabled = True

    if supported_models_checklist is not None and 'specific_supported_model' in supported_models_checklist:
        supported_models_dropdown_disabled = False
        if supported_models_dropdown is not None:
            supported_model = supported_models_dropdown
            if isinstance(supported_model, list):
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
    # print(result_df_restricted.index)
    summary_df_restricted_tree, eligible_points_df_restricted, _ = get_details(result_df_restricted, verbose=False)
    p_txt = get_text_stats(eligible_points_df_restricted)
    fig = pareto(summary_df_restricted_tree, show=False)
    output = (
        {'display': 'none'} if supported_models_dropdown_disabled else {'display': 'block'},
        {'display': 'none'} if required_outputs_dropdown_disabled else {'display': 'block'},
        fig,
        p_txt,
        url,
    )
    return output


if __name__ == '__main__':
    app.run_server(debug=False, port=8005)  # dev_tools_hot_reload=False,
