class Test:
    # todo add last_update = version of the release
    description = None
    input_features = []
    dataset_size = None
    X = None
    dataset_to_explain = None
    trained_model = None
    predict_func = None

    def score(self, **kwargs):
        # todo assert attribution_values feature_importance size
        raise NotImplementedError("The scoring method of this test is not implemented at the moment.")
