from src.utils import are_class_names_unique
from tests import *

valid_tests = (
    DetectInteraction0,
    DetectInteraction1,
    DetectInteraction2,
    DetectInteraction3,
    DetectInteraction4,
    # CoughAndFever,
    # Mnist,
    # CoughAndFever1090,
    # DistributionNonUniformStatDep,
    # DistributionUniformStatDep,
    # DistributionNonUniformStatIndep,
)
assert are_class_names_unique(valid_tests), 'Duplicate test names'
valid_tests_dico = {e.name: e for e in valid_tests}

# not working yet
# "faithfulness": tests.Faithfulness,
# "roar_faithfulness": tests.ROARFaithfulness,
# "roar_monotonicity": tests.ROARMonotonicity,
# "monotonicity": tests.Monotonicity,
# "roar": tests.Roar,
# "shapley": tests.Shapley,
# "shapley_corr": tests.ShapleyCorr,
# "infidelity": tests.Infidelity
