import numpy as np
import pandas as pd

from .src.io import load_results
from .src.explainer import valid_explainers_dico
from .src.export_sql import export_to_sql

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

























    # For appendix
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





    #############################
    explainer_table = explainer[explainer.is_implemented == "1"].copy()
    # explainer_table = explainer_table[explainer_table.explainer != "archipelago"]
    # explainer_table = explainer_table[explainer_table.explainer != "shap_interaction"]
    # explainer_table = explainer_table[explainer_table.explainer != "shapley_taylor_interaction"]
    explainer_table.explainer = '\href{' + explainer_table.implementation_link + '}{' + explainer_table.explainer.str.replace('_', '\\_') + '}'

    def f(supported_model_model_agnostic, supported_model_tree_based, supported_model_neural_network):
        if supported_model_model_agnostic:
            return 'The xAI algorithm is model agnostic i.e. it can explain any AI model.'

        s = 'The xAI algorithm can explain'
        if supported_model_tree_based:
            s+=' tree-based models'
            if supported_model_neural_network:
                s += ' and neural networks'
            s += '.'
            return s
        if supported_model_neural_network:
            s += ' neural networks.'
            return s
    explainer_table['supported_model_str'] = [
        f(a,b,c) for a,b,c in zip(explainer_table.supported_model_model_agnostic,
                                  explainer_table.supported_model_tree_based,
                                  explainer_table.supported_model_neural_network,)]

    def f(output_attribution,output_importance,output_interaction):
        out = []
        if output_attribution:
            out+=[output_labels['attribution']]
        if output_importance:
            out+=[output_labels['importance']]
        if output_interaction:
            out+=[output_labels['interaction']]
        return ', '.join(out)

    explainer_table['output_str'] = [
        f(output_attribution, output_importance, output_interaction)
        for output_attribution,output_importance,output_interaction
        in zip(explainer_table.output_attribution,explainer_table.output_importance,explainer_table.output_interaction)
    ]


    def f(source_paper_tag):
        if pd.isna(source_paper_tag) or source_paper_tag=='-':
            return ''
        else:
            return ' \citep{' + source_paper_tag + '}'

    explainer_table.source_paper_tag = explainer_table.source_paper_tag.apply(f)

    def f(required_input_data):
        if pd.isna(required_input_data) or required_input_data=='\n\t\t\t - ':
            return ''
        return 'The following information are required by the xAI algorithm: ' + required_input_data.replace('-', ',').replace('_', ' ')
    explainer_table.required_input_data = explainer_table.required_input_data.apply(f)


    explainer_table['annex'] = '\n\\item[' + explainer_table.explainer + '] '
    explainer_table['annex'] += '\n' + explainer_table.source_paper_tag
    explainer_table['annex'] += ' \n' + explainer_table.description
    explainer_table['annex'] += ' \n' + explainer_table.supported_model_str
    explainer_table['annex'] += ' \nThe xAI algorithm can output the following explanations: ' + explainer_table['output_str'] + '.'
    explainer_table['annex'] += ' \n' + explainer_table.required_input_data



    tex = '\\begin{description}\n\n' + explainer_table.annex.str.cat(sep='\n') + '\n\n\end{description}'
    with open("explainer.tex", "w") as f:
            f.write(tex)
