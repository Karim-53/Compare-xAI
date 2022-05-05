from src.dask.utils import load_results
import pandas as pd
import numpy as np

from src.explainer import valid_explainers_dico
def get_required_input_data(explainer_name) -> set:
    if explainer_name not in valid_explainers_dico.keys():
        print(f'{explainer_name}  not in valid_explainers_dico.keys()')
        return None
    explainer_class = valid_explainers_dico[explainer_name]
    _init_specific_args = explainer_class.get_specific_args_init()
    _explain_specific_args = explainer_class.get_specific_args_explain()
    return _init_specific_args.union(_explain_specific_args)


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
    print(cross_tab)
    print('writing files to /data/03_experiment_output_aggregated/ ...')
    cross_tab.to_parquet('../data/03_experiment_output_aggregated/cross_tab.parquet')
    cross_tab.to_csv('../data/03_experiment_output_aggregated/cross_tab.csv')

    ############################################################################################################
    explainer = pd.read_csv('../data/01_raw/explainer.csv')
    for out in ['attribution', 'importance', 'interaction']:
        explainer[f'output_{out}'] = [None if pd.isna(s) else float(out in eval(s)) for s in explainer.outputs]

    for x in ['model_agnostic', 'tree_based', 'neural_network']:
        explainer[f'supported_model_{x}'] = [None if pd.isna(s) else x in eval(s) for s in explainer.supported_models]
        explainer[f'supported_model_{x}'] = np.logical_or(explainer[f'supported_model_{x}'],
                                                          explainer.supported_model_model_agnostic)

    explainer['required_input_data'] = [get_required_input_data(explainer_name) for explainer_name in
                                        explainer.explainer]
    all_required_input_data = set().union(*list(explainer[explainer.required_input_data.notna()].required_input_data))
    print('all_required_input_data', all_required_input_data)
    for required_input_data in all_required_input_data:
        explainer[f'required_input_{required_input_data}'] = [None if pd.isna(s) else required_input_data in s for s in
                                                              explainer.required_input_data]
    explainer.required_input_X_reference = np.logical_or(explainer.required_input_X,
                                                         explainer.required_input_X_reference)  # todo fix that in prior code

    print(explainer)
    print('writing explainer to /data/03_experiment_output_aggregated/ ...')
    explainer.to_parquet('../data/03_experiment_output_aggregated/explainer.parquet')
    explainer.to_csv('../data/03_experiment_output_aggregated/explainer.csv')
