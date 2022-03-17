import shap
import explainers

valid_explainers = {
    "treeshap": explainers.TreeShap,
    # "shap": explainers.Shap,
    # "shapr": explainers.ShapR,
    "kernelshap": explainers.KernelShap,
    # "brutekernelshap": explainers.BruteForceKernelShap,
    # "random": explainers.Random,
    "lime": explainers.Lime,
    # "maple": explainers.Maple,

    # "l2x": explainers.L2X,
    # "breakdown": explainers.BreakDown,
}


class Explainer:
    def __init__(self, name, **kwargs):
        if name not in valid_explainers.keys():
            raise NotImplementedError(
                f"This explainer is not supported at the moment. Explainers supported are {list(valid_explainers.keys())}"
            )
        self.name = name
        self.explainer = lambda trained_model, data: valid_explainers[name](trained_model, data, **kwargs)
