import xgboost as xgb
from xgboost import XGBRegressor

from explainers.explainer_superclass import Explainer
from src.utils import get_feature_importance


class Saabas(Explainer):
    name = 'saabas'
    # requirements = {'generic_xgboost':True}

    expected_values = None
    attribution_values = None
    feature_importance = None

    def __init__(self, trained_model, **kwargs):
        self.trained_model = trained_model
        if isinstance(trained_model, XGBRegressor):  # or classifier
            self.trained_model = trained_model.get_booster()

    def explain(self, dataset_to_explain, **kwargs):
        dmatrix_to_explain = xgb.DMatrix(dataset_to_explain)
        saabas_values_3 = self.trained_model.predict(dmatrix_to_explain, pred_contribs=True, approx_contribs=True)
        self.attribution_values = saabas_values_3[:, :-1]

        # self.expected_values = shap_values.base_values  # todo [after acceptance] get the base value from the xgboost
        self.feature_importance = get_feature_importance(self.attribution_values)

# todo if random forest :
# from treeinterpreter import treeinterpreter as ti, utils
# from sklearn.ensemble import RandomForestRegressor
#
# rf = RandomForestRegressor(n_estimators=3, max_depth=3, bootstrap=False)
#
# rf.fit(df_train[input_features].values, df_train['target'])
# prediction, bias, contributions = ti.predict(rf, df_train[input_features].values, joint_contribution=True)
# aggregated_contributions = utils.aggregated_contribution(contributions)
