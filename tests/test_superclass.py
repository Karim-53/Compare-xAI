class Test:
    # todo add last_update = version of the release
    description = None
    input_features = []
    dataset_size = None  # deprecated to delete
    X = None  # must be 2d (nb of samples, nb of features)
    dataset_to_explain = None  # must be 2d (nb of samples, nb of features)
    trained_model = None
    predict_func = None

    # todo add init with var check

    def score(self, attribution_values=None, feature_importance=None, **kwargs):
        # todo assert attribution_values feature_importance size
        raise NotImplementedError("The scoring method of this test is not implemented at the moment.")
