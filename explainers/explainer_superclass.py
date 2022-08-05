import re

import numpy as np
import pandas as pd

from src.utils import root

MODELS = ['tree_based', 'neural_network']
EXTENDED_MODELS = {'model_agnostic': MODELS}

import inspect


def supported_models_developed(supported_models):
    _supported_models_developed = list(supported_models)
    for e in supported_models:
        _supported_models_developed += EXTENDED_MODELS.get(e, [])
    return _supported_models_developed


def _len(x):
    if x is None:
        return 0
    return len(x)


def get_specific_args(method, super_method) -> set:
    arg_spec = inspect.getfullargspec(method)
    args = arg_spec.args
    len_default_args = _len(arg_spec.defaults)
    if len_default_args:  # keep only required args
        args = args[:-len_default_args]
    super_method_init_args = inspect.getfullargspec(super_method).args
    return set(args).difference(set(super_method_init_args))


regex_to_snake_case = re.compile(
    r'(?<!^)(?=[A-Z])')  # for a more advanced case: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
to_snake_case = lambda name: regex_to_snake_case.sub('_', name).lower()
explainer_csv = pd.read_csv(root + '/data/01_raw/explainer.csv').set_index('explainer')


class Explainer:
    """ The doc is different from description prop:
    doc is for coders while description is for the end user data scientist seeing the website"""
    # todo change this to shap 's format: use __call__ ect. (literally this one should inherit from that class)
    name: str  # will be inferred from class name
    supported_models = ()  # supported in the implementation not in theory # todo [after acceptance] add supported_models_theory if needed in filters

    # Know what could be calculated
    output_importance = False
    output_attribution = False
    output_interaction = False

    # xAI output
    importance: np.array = None
    attribution: np.ndarray = None
    interaction: np.ndarray = None

    supported_model_model_agnostic: bool
    supported_model_tree_based: bool
    supported_model_neural_network: bool

    description = None  # if the xai pretend to be the unique solution given these assumptions / axioms please write it here until I find a way to index it

    score_time_dominate = None
    score_time_dominated_by = None
    score_dominate = None
    score_dominated_by = None

    source_code = None
    source_paper_tag = None
    source_paper_bibliography = None

    # todo [after acceptance] add complexity as str, is_affected_by_seed
    # todo [after acceptance] add last_update = version of the release of this repo
    # todo add source paper just the bibtex tag
    # todo add a pretty way to print the class
    def __init_subclass__(cls, name=None, **kwargs):  # todo delete name param and replace it by cls.__name__
        super().__init_subclass__(**kwargs)
        if name is None:
            # print(f'Warning: {cls.__name__} Explainer defined without a name property')
            name = to_snake_case(cls.__name__)
        cls.name = name
        if cls.name in explainer_csv.index:
            row = explainer_csv.loc[cls.name]
            for attribute, val in row.items():
                if ' ' not in attribute:
                    setattr(cls, attribute, val)
        else:
            print(f'Warning: {cls.name} Explainer is not mentioned in explainer.csv')

        # self.source_paper_bibliography = bibliography.get(self.source_paper_tag, None)

    @classmethod
    def get_xai_output(cls):
        xai_output = []
        for out, out_str in zip(['importance', 'attribution', 'interaction'],
                                ['feature importance', 'feature attribution', 'pair interaction']):
            if cls.__dict__.get(out, False):
                xai_output.append(out_str)
        return xai_output

    @classmethod
    def get_specific_args_init(cls) -> set:
        return get_specific_args(cls.__init__, Explainer.__init__)

    @classmethod
    def get_specific_args_explain(cls) -> set:
        return get_specific_args(cls.explain, Explainer.explain)

    def __repr__(self) -> str:  # it must be a string
        return self.to_pandas().__repr__()  # todo fix add string saying that it is an instance otherwise it is gonna be confusing

    @classmethod
    def to_pandas(cls) -> pd.Series:  # todo add params include_dominance, include_results
        d = {}
        d['name'] = cls.name
        d['supported_models'] = cls.supported_models

        d['xai_s_output'] = cls.get_xai_output()
        d['.__init__() specific args'] = cls.get_specific_args_init()
        d['.explain() specific args'] = cls.get_specific_args_explain()
        d['description'] = cls.description

        d['source_paper'] = cls.source_paper_bibliography
        d['source_code'] = cls.source_code

        df = pd.Series(d, name=cls.name)
        return df

    def __str__(self) -> str:
        xai_output = self.get_xai_output()

        method_init_specific_args = self.get_specific_args_init()
        method_explain_specific_args = self.get_specific_args_explain()

        s = f"""{self.__class__.__name__}(Explainer)
name:\t\t\t\t{self.name}
supported_models:\t{self.supported_models}
xAI's output:\t\t {', '.join(xai_output)} 
"""
        s += f"\n.__init__() specific args: {', '.join(method_init_specific_args)}" if len(
            method_init_specific_args) else ''

        s += f"\n.explain()  specific args: {', '.join(method_explain_specific_args)}" if len(
            method_explain_specific_args) else ''

        s += f"\ndescription:\t{self.description}" if self.description is not None else ''
        return s

    def explain(self,  # todo [after acceptance] think about changing this to __call__ ?
                dataset_to_explain,
                # todo [after acceptance] think about changing var name to explicand == the input to be explained see https://arxiv.org/pdf/1908.08474.pdf
                **kwargs):  # Should we make this an abstract method ?
        """ Here we should cite 1 or 2 sentences why the xai need such variables """
        # todo create a decorator for this function: automatically infer global explanation and check the output format
        from src.explainer import valid_explainers
        raise NotImplementedError(
            f"This explainer is not supported at the moment. Explainers supported are {[e.name for e in valid_explainers]}"
        )

    def check_explanation(self, dataset_to_explain):  # , **kwargs
        arr = np.array(dataset_to_explain)
        _shape = arr.shape
        if len(_shape) == 1:
            _shape = (1, _shape[0])
        # self.expected_values = np.random.randn(_shape[0])

        for var_name, expected_shape in [['attribution', _shape], ['importance', (
                _shape[1],)]]:  # todo self.interaction = np.random.randn(_shape[1], _shape[1])
            var = self.__dict__.get(var_name)
            if var is not None and not isinstance(var, str):
                if var.shape != expected_shape:  # todo also verify it is numpy
                    print(f'{self.name} {var_name}: Wrong shape. received {var.shape}  should be {expected_shape}')

    # todo [after acceptance] think about how to sort explainer instances by name or expected execution time https://stackoverflow.com/questions/4010322/sort-a-list-of-class-instances-python


class InteractionExplainer:
    def __init__(
            self,
            model,
            input=None,
            baseline=None,
            # todo ake this a parameter in the tests a part, because it is frequently use in atchipelago and in https://arxiv.org/pdf/1908.08474.pdf [18, 2, 10].
            data_xformer=None,
            output_indices=0,
            batch_size=20,
            verbose=False,
    ):

        input, baseline = self.arg_checks(input, baseline, data_xformer)

        self.model = model
        self.input = np.squeeze(input)
        self.baseline = np.squeeze(baseline)
        self.data_xformer = data_xformer
        self.output_indices = output_indices
        self.batch_size = batch_size
        self.verbose = verbose

    def arg_checks(self, input, baseline, data_xformer):
        if (input is None) and (data_xformer is None):
            raise ValueError("Either input or data xformer must be defined")

        if input is not None and baseline is None:
            raise ValueError("If input is defined, the baseline must also defined")

        if data_xformer is not None and input is None:
            input = np.ones(data_xformer.num_features).astype(bool)
            baseline = np.zeros(data_xformer.num_features).astype(bool)
        return input, baseline

    def verbose_iterable(self, iterable):
        if self.verbose:
            from tqdm import tqdm

            return tqdm(iterable)
        else:
            return iterable

    def batch_set_inference(
            self, set_indices, context, insertion_target, include_context=False
    ):
        """
        Creates archipelago type data instances and runs batch inference on them
        All "sets" are represented as tuples to work as keys in dictionaries
        """

        num_batches = int(np.ceil(len(set_indices) / self.batch_size))

        scores = {}
        for b in self.verbose_iterable(range(num_batches)):
            batch_sets = set_indices[b * self.batch_size: (b + 1) * self.batch_size]
            data_batch = []
            for index_tuple in batch_sets:
                new_instance = context.copy()
                for i in index_tuple:
                    new_instance[i] = insertion_target[i]

                if self.data_xformer is not None:
                    new_instance = self.data_xformer(new_instance)

                data_batch.append(new_instance)

            if include_context and b == 0:
                if self.data_xformer is not None:
                    data_batch.append(self.data_xformer(context))
                else:
                    data_batch.append(context)

            preds = self.model(np.array(data_batch))

            for c, index_tuple in enumerate(batch_sets):
                scores[index_tuple] = preds[c, self.output_indices]
            if include_context and b == 0:
                context_score = preds[-1, self.output_indices]

        output = {"scores": scores}
        if include_context and num_batches > 0:
            output["context_score"] = context_score
        return output


class Random(Explainer, name='baseline_random'):
    description = 'This is not a real explainer it helps measure the baseline score and processing time.'
    supported_models = ('model_agnostic',)
    output_attribution = True
    output_importance = True
    output_interaction = True

    source_paper_tag = 'liu2021synthetic'
    source_paper_bibliography = r"""@article{liu2021synthetic,
  title={Synthetic benchmarks for scientific research in explainable machine learning},
  author={Liu, Yang and Khandagale, Sujay and White, Colin and Neiswanger, Willie},
  journal={arXiv preprint arXiv:2106.12543},
  year={2021}"""
    source_code = 'https://github.com/abacusai/xai-bench'

    def __init__(self, **kwargs):
        super().__init__()

    def explain(self, dataset_to_explain, **kwargs):
        # todo [after acceptance] with np seed = 0
        arr = np.array(dataset_to_explain)
        _shape = arr.shape
        if len(_shape) == 1:
            _shape = (1, _shape[0])
        self.expected_values = np.random.randn(_shape[0])
        self.attribution = np.random.randn(*_shape)
        self.importance = np.random.randn(_shape[1])

        self.interaction = np.random.randn(_shape[1], _shape[1])


class UnsupportedModelException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


if __name__ == '__main__':
    explainer = Random()
    print(explainer.supported_model_neural_network)
