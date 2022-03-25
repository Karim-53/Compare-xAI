import numpy as np

MODELS = ['tree_based', 'neural_network']
EXTENDED_MODELS = {'model_agnostic': MODELS}


def supported_models_developed(supported_models):
    _supported_models_developed = list(supported_models)
    for e in supported_models:
        _supported_models_developed += EXTENDED_MODELS.get(e, [])
    return _supported_models_developed


class Explainer:
    name = None
    description = None
    supported_models = ()

    # todo [after acceptance] add complexity as str, is_affected_by_seed
    # todo [after acceptance] add last_update = version of the release of this repo
    # todo add source paper just the bibtex tag
    # todo add a pretty way to print the class

    # Know what could be calculated
    attribution = False
    importance = False
    interaction = False

    def explain(self, x, **kwargs):  # todo [after acceptance] change this to __call__ ?
        from src.explainer import valid_explainers
        raise NotImplementedError(
            f"This explainer is not supported at the moment. Explainers supported are {[e.name for e in valid_explainers]}"
        )


class InteractionExplainer:
    def __init__(
            self,
            model,
            input=None,
            baseline=None,
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


class Random(Explainer):
    name = 'baseline_random'
    description = 'This is not a real explainer it helps measure the baseline score and processing time.'
    supported_models = ('model_agnostic',)
    attribution = True
    importance = True
    interaction = True

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
