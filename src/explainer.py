import shap
from explainers import *

valid_explainers = (
    Saabas,
    Random,
    # Maple,
    TreeShap,
    # KernelShap,
    # Lime,
)
# not working yet
# BreakDown,  # need cpp14
# "shap": explainers.Shap,
# "shapr": explainers.ShapR,
# "brutekernelshap": explainers.BruteForceKernelShap,
# "l2x": explainers.L2X,
# "random": explainers.Random, # todo delete
