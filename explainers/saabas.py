import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import sys
import traceback

from explainers.explainer_superclass import Explainer, UnsupportedModelException
from src.utils import get_importance


class Saabas(Explainer):
    name = 'saabas'
    supported_models = ('tree_based',)
    # requirements = {'generic_xgboost':True}

    output_attribution = True
    output_importance = True  # inferred

    def __init__(self, trained_model, **kwargs):
        super().__init__()
        self.trained_model = trained_model
        try:
            if isinstance(trained_model, XGBRegressor):  # or classifier
                self.trained_model = trained_model.get_booster()
        except:
            exc_info = sys.exc_info()
            traceback.print_exception(*exc_info)
            raise UnsupportedModelException()
        if not isinstance(self.trained_model, xgb.core.Booster) and not isinstance(self.trained_model, RandomForestRegressor):
            raise UnsupportedModelException()

    def explain(self, dataset_to_explain, **kwargs):
        self.interaction = None
        if isinstance(self.trained_model, xgb.core.Booster):
            dmatrix_to_explain = xgb.DMatrix(
                dataset_to_explain)  # dataset_to_explain must be pd.DataFrame otherwise error
            saabas_values_3 = self.trained_model.predict(dmatrix_to_explain, pred_contribs=True, approx_contribs=True)
            self.attribution = saabas_values_3[:, :-1]

            # self.expected_values = shap_values.base_values  # todo [after acceptance] get the base value from the xgboost
            self.importance = get_importance(self.attribution)
        elif isinstance(self.trained_model, RandomForestRegressor):
            raise 'not implemented' # todo if random forest
            # from treeinterpreter import treeinterpreter as ti, utils
            # from sklearn.ensemble import RandomForestRegressor
            #
            # rf = RandomForestRegressor(n_estimators=3, max_depth=3, bootstrap=False)
            #
            # rf.fit(df_train[input_features].values, df_train['target'])
            # prediction, bias, contributions = ti.predict(rf, df_train[input_features].values, joint_contribution=True)
            # aggregated_contributions = utils.aggregated_contribution(contributions)
        else:
            print('####### Saabas works only with tree-based model #', type(self.trained_model))
            raise UnsupportedModelException()


if __name__ == '__main__':
    test = Saabas(trained_model=lambda x: x)
    print(test)
