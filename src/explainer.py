from explainers import *
from src.utils import are_class_names_unique

valid_explainers = (
    Random,  # keep on top
    Permutation,
    PermutationPartition,
    Partition,
    TreeShapApproximation,
    ExactShapleyValues,
    TreeShap,
    Saabas,
    KernelShap,
    Sage,
    Lime,
    Maple,
    JointShapley,
    # Anova, # todo fix feature importance

    # Archipelago,  # todo after paper acceptance
    # ShapleyTaylorInteraction,  # todo after paper acceptance
    # ShapInteraction,  # todo after paper acceptance
)
assert are_class_names_unique(valid_explainers), 'Duplicate explainer names'

valid_explainers_dico = {e.name: e for e in valid_explainers}
# not working yet
# BreakDown,  # need cpp14
# "shap": explainers.Shap,
# "shapr": explainers.ShapR,
# "brutekernelshap": explainers.BruteForceKernelShap,
# "l2x": explainers.L2X,

if __name__ == '__main__':
    pass
