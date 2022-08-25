import numpy as np
import pandas as pd

from src.io import load_results
from src.explainer import valid_explainers_dico
from src.export_sql import export_to_sql

# todo delete supported_models and outputs columns
# todo generate supported_models and outputs columns for the detailed view of an explainer

required_data_to_nice_name = {
    'trained_model': "AI model's structure",
    'ml_task': 'Nature of the ML task (regression/classification)',

    'predict_proba': "The model's predict probability function",
    'predict_func': "The model's predict function",

    'df_reference': 'A reference dataset',
    'X_reference': 'A reference dataset (input only)',
    'X': 'The train set',

    'truth_to_explain': 'True output of the data points to explain',
}


def get_required_input_data(explainer_name) -> set:
    # print('valid explainers:', len(valid_explainers_dico), valid_explainers_dico.keys())
    assert len(valid_explainers_dico) > 5
    if explainer_name not in valid_explainers_dico.keys():
        print(f'{explainer_name}  not in valid_explainers_dico.keys()')
        return None
    explainer_class = valid_explainers_dico[explainer_name]
    _init_specific_args = explainer_class.get_specific_args_init()
    _explain_specific_args = explainer_class.get_specific_args_explain()
    return _init_specific_args.union(_explain_specific_args)


output_labels = {'importance': 'Feature importance (global explanation)',
                 'attribution': 'Feature attribution (local explanation)',  # can use html here
                 'interaction': 'Feature interaction (local explanation)',
                 }
bullet_point = '\n\t\t\t - '


def aggregate_outputs(explainer: pd.DataFrame):
    _df = pd.DataFrame()
    for out, label in output_labels.items():
        _df[out] = explainer[f'output_{out}'].map({True: label, False: None})
    return _df.apply(lambda x: bullet_point + bullet_point.join(x.dropna().values.tolist()), axis=1)


def aggregate_supported_models(explainer: pd.DataFrame):
    _df = pd.DataFrame()
    model_labels = {
        'model_agnostic': 'Any AI model (model agnostic xAI algorithms are independent of the AI implementation)',
        'tree_based': 'Tree-based ML',
        'neural_network': 'Neural networks',
    }
    for key, label in model_labels.items():
        _df[key] = explainer[f'supported_model_{key}'].map({True: label, False: None})
    return _df.apply(lambda x: bullet_point + bullet_point.join(x.dropna().values.tolist()), axis=1)


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


    def subtest_to_tested_xai_output(subtest):
        for out, label in output_labels.items():
            if out in subtest:
                return label
        print('fix me in aggregate_data.py')
        return None


    cross_tab['tested_xai_output'] = cross_tab.subtest.apply(subtest_to_tested_xai_output)
    # print(cross_tab)
    print('writing files to /data/03_experiment_output_aggregated/ ...')
    cross_tab.to_parquet('../data/03_experiment_output_aggregated/cross_tab.parquet')
    cross_tab.to_csv('../data/03_experiment_output_aggregated/cross_tab.csv', index=False)

    ############################################################################################################
    explainer = pd.read_csv('../data/01_raw/explainer.csv')
    explainer.sort_values('explainer', inplace=True)
    # todo delete supported_models column from 01_raw/explainer.csv because we add it now
    # todo delete outputs column from 01_raw/explainer.csv because we add it now

    explainer['outputs'] = aggregate_outputs(explainer)

    for x in ['model_agnostic', 'tree_based', 'neural_network']:
        explainer[f'supported_model_{x}'] = np.logical_or(explainer[f'supported_model_{x}'],
                                                          explainer.supported_model_model_agnostic)

    explainer['supported_models'] = aggregate_supported_models(explainer)

    explainer['required_input_data'] = [get_required_input_data(explainer_name) for explainer_name in
                                        explainer.explainer]
    all_required_input_data = set(
        sorted(set().union(*list(explainer[explainer.required_input_data.notna()].required_input_data))))
    print('all_required_input_data', all_required_input_data)
    for required_input_data in all_required_input_data:
        explainer[f'required_input_{required_input_data}'] = [None if pd.isna(s) else required_input_data in s for s in
                                                              explainer.required_input_data]
    if 'X' in all_required_input_data and 'X_reference' in all_required_input_data:
        explainer.required_input_X_reference = np.logical_or(explainer.required_input_X,
                                                             explainer.required_input_X_reference)  # todo fix that in prior code

    explainer.required_input_data = explainer.required_input_data.apply(
        lambda _set: None if _set is None else bullet_point + bullet_point.join([required_data_to_nice_name.get(e,e) for e in _set])
    )

    time_per_test = cross_tab[['explainer', 'time']].groupby('explainer').mean()
    time_per_test = time_per_test.rename(columns={'time': 'time_per_test'})
    time_per_test['explainer'] = time_per_test.index
    explainer = explainer.join(time_per_test, on='explainer', rsuffix='_to_drop').drop('explainer_to_drop', axis=1)
    # print('todo required_input_train_function is set to 0')
    # explainer['required_input_train_function'] = 0
    print('writing explainer to /data/03_experiment_output_aggregated/ ...')
    explainer.to_parquet('../data/03_experiment_output_aggregated/explainer.parquet')
    explainer.to_csv('../data/03_experiment_output_aggregated/explainer.csv', index=False)
    export_to_sql()
    print('End')

    ############################################################################################################
    test = pd.read_csv('../data/01_raw/test.csv')
    test = test[test.is_shortlisted == 1]
    test = test[test.is_implemented == "1"]
    test.to_csv('../data/03_experiment_output_aggregated/test.csv', index=False)

    # supplementary material
    test_table = test[~test.test.str.contains('detect_interaction')].copy()

    for col in ['description', 'test_procedure', 'test_metric', 'category_justification']:
        test_table['end'] = test_table[col].str[-1]
        _summary = test_table[test_table.end!='.'][['test','end']]
        if len(_summary):
            print(col)
            print(_summary)




    test_table['test'] = '\href{' + test_table.test_implementation_link + '}{' + test_table.test.str.replace('_', '\\_') + '}'
    def f(dataset, dataset_source):
        if dataset_source == '-':
            return 'a ' + dataset.replace('_', ' ')
        else:
            return 'the \href{' + dataset_source + '}{' + dataset.replace('_', ' ') + '}'


    test_table['dataset'] = [f(dataset, dataset_source) for dataset, dataset_source in zip(test_table.dataset,test_table.dataset_source)]
    test_table['model'] = test_table.model.replace('function', 'a custom function')
    test_table['model'] = test_table.model.replace('MLP', 'an MLP')

    # test_table.category_justification = 'p{'+test_table.category_justification+'}'
    # test_table.short_description = 'p{'+test_table.short_description+'}'
    # test_table.test_procedure = 'p{'+test_table.test_procedure+'}'
    # test_table.description = 'p{'+test_table.description+'}'
    # test_table.test_metric = 'p{'+test_table.test_metric+'}'

    columns = ['test',
               'short_description', 'description',
               'category', 'category_justification',
               'dataset', 'dataset_size', 'model',
               'test_procedure',
               'test_metric',
               ]
    test_table = test_table[columns] # .str.replace('%',' percent')
    # test_table.columns = test_table.columns.str.title().str.replace('_', ' ')
    # with pd.option_context("max_colwidth", 3000):
    #     tex = test_table.to_latex(index=False, escape=False, na_rep='')
    #
    # with open("test_table.tex", "w") as f:
    #         f.write(tex)


    test_table['annex'] = '\n\n\\item['+test_table.test+'] answers the following question: \\emph{' + test_table.short_description + '}.'
    test_table['annex'] += '\n'+ test_table.description  # nzid point
    test_table['annex'] += '\n'+ ' The test utilize \\textbf{'+test_table.model+'} model trained on \\textbf{'+test_table.dataset +'} dataset (total size: '+test_table.dataset_size.astype('int').astype('str') +').'
    test_table['annex'] += '\n The test procedure is as follows: ' + test_table.test_procedure
    test_table['annex'] += '\n The score is calculated as follows: ' + test_table.test_metric
    test_table['annex'] += '\n'+ ' The test is classified in the \\textbf{'+test_table.category+'} category because ' + test_table.category_justification   # nzid point

    tex = '\\begin{description}\n\n' + test_table.annex.str.cat(sep='\n') + '\n\n\end{description}'

    with open("test.tex", "w") as f:
            f.write(tex)

