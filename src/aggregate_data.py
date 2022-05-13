import numpy as np
import pandas as pd

from src.dask.utils import load_results
from src.explainer import valid_explainers_dico
from src.export_sql import export_to_sql


# todo delete supported_models and outputs columns
# todo generate supported_models and outputs columns for the detailed view of an explainer


def get_required_input_data(explainer_name) -> set:
    print('valid explainers:', len(valid_explainers_dico), valid_explainers_dico.keys())
    if explainer_name not in valid_explainers_dico.keys():
        print(f'{explainer_name}  not in valid_explainers_dico.keys()')
        return None
    explainer_class = valid_explainers_dico[explainer_name]
    _init_specific_args = explainer_class.get_specific_args_init()
    _explain_specific_args = explainer_class.get_specific_args_explain()
    return _init_specific_args.union(_explain_specific_args)


def aggregate_outputs(explainer: pd.DataFrame):
    _df = pd.DataFrame()
    output_labels = {'importance': 'Feature importance (global explanation)',
                     'attribution': 'Feature attribution (local explanation)',  # can use html here
                     'interaction': 'Feature interaction (local explanation)',
                     }
    for out, label in output_labels.items():
        _df[out] = explainer[f'output_{out}'].map({True: label, False: None})
    return _df.apply(lambda x: ','.join(x.dropna().values.tolist()), axis=1)


def aggregate_supported_models(explainer: pd.DataFrame):
    _df = pd.DataFrame()
    model_labels = {
        'model_agnostic': 'Any AI model (model agnostic xAI algorithms are independent of the AI implementation)',
        'tree_based': 'Tree-based ML',
        'neural_network': 'Neural networks',
        }
    for key, label in model_labels.items():
        _df[key] = explainer[f'supported_model_{key}'].map({True: label, False: None})
    return _df.apply(lambda x: ','.join(x.dropna().values.tolist()), axis=1)


if __name__ == "__main__":
    result_df = load_results()
    lista = []
    for column_name, column in result_df.iteritems():
        for explainer, dico in column.items():
            print(explainer, dico)
            time = dico.get('time', None)
            last_updated = dico.get('Last_updated', None)
            for subtest, score in dico.get('score', {}).items():
                if pd.isna(score):
                    score = None
                lista.append({'explainer': explainer,
                              'test': column_name,
                              'subtest': subtest,
                              'test_subtest': column_name + '_' + subtest,
                              'score': score,
                              'time': time,
                              'last_updated': last_updated,
                              })
    cross_tab = pd.DataFrame(lista).sort_values(['explainer', 'test', 'subtest'])
    # print(cross_tab)
    print('writing files to /data/03_experiment_output_aggregated/ ...')
    cross_tab.to_parquet('../data/03_experiment_output_aggregated/cross_tab.parquet')
    cross_tab.to_csv('../data/03_experiment_output_aggregated/cross_tab.csv', index=False)

    ############################################################################################################
    explainer = pd.read_csv('../data/01_raw/explainer.csv')
    # todo delete supported_models column from 01_raw/explainer.csv because we add it now
    # todo delete outputs column from 01_raw/explainer.csv because we add it now

    explainer['outputs'] = aggregate_outputs(explainer)

    for x in ['model_agnostic', 'tree_based', 'neural_network']:
        explainer[f'supported_model_{x}'] = np.logical_or(explainer[f'supported_model_{x}'],
                                                          explainer.supported_model_model_agnostic)

    explainer['supported_models'] = aggregate_supported_models(explainer)

    explainer['required_input_data'] = [get_required_input_data(explainer_name) for explainer_name in
                                        explainer.explainer]
    all_required_input_data = set().union(*list(explainer[explainer.required_input_data.notna()].required_input_data))
    print('all_required_input_data', all_required_input_data)
    for required_input_data in all_required_input_data:
        explainer[f'required_input_{required_input_data}'] = [None if pd.isna(s) else required_input_data in s for s in
                                                              explainer.required_input_data]
    if 'X' in all_required_input_data and 'X_reference' in all_required_input_data:
        explainer.required_input_X_reference = np.logical_or(explainer.required_input_X,
                                                             explainer.required_input_X_reference)  # todo fix that in prior code

    explainer.required_input_data = explainer.required_input_data.apply(str)
    # print('todo required_input_train_function is set to 0')
    # explainer['required_input_train_function'] = 0
    print(explainer)
    print('writing explainer to /data/03_experiment_output_aggregated/ ...')
    explainer.to_parquet('../data/03_experiment_output_aggregated/explainer.parquet')
    explainer.to_csv('../data/03_experiment_output_aggregated/explainer.csv', index=False)
    export_to_sql()
    print('End')
