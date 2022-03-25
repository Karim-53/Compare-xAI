from explainers import *
from src.utils import are_class_names_unique

valid_explainers = (
    Random,
    ShapInteraction,
    Saabas,
    TreeShap,
    Sage,
    Lime,
    KernelShap,
    Maple,
)
assert are_class_names_unique(valid_explainers), 'Duplicate explainer names'

valid_explainers_dico = {e.name:e for e in valid_explainers}
# not working yet
# BreakDown,  # need cpp14
# "shap": explainers.Shap,
# "shapr": explainers.ShapR,
# "brutekernelshap": explainers.BruteForceKernelShap,
# "l2x": explainers.L2X,
# "random": explainers.Random, # todo delete
