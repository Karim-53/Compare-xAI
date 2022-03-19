import shap
from explainers import *

valid_explainers = (
    Random,
    # Maple,
    TreeShap,
    # KernelShap,
    # Lime,
)
# not working yet
# "shap": explainers.Shap,
# "shapr": explainers.ShapR,
# "brutekernelshap": explainers.BruteForceKernelShap,
# "l2x": explainers.L2X,
# "breakdown": explainers.BreakDown,
# "random": explainers.Random, # todo delete
