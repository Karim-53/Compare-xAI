from src.utils import are_class_names_unique
from tests import *

valid_tests = (
    CoughAndFever,
    CoughAndFever1090,
    AAndBOrC,
    DistributionNonUniformStatDep,
    DistributionUniformStatDep,
    DistributionNonUniformStatIndep,
    DetectInteraction0,
    DetectInteraction1,
    DetectInteraction2,
    DetectInteraction3,
    DetectInteraction4,
    Mnist,
)
assert are_class_names_unique(valid_tests), 'Duplicate test names'
valid_tests_dico = {e.name: e for e in valid_tests}


def get_sub_tests(test_name):
    test_instance = valid_tests_dico[test_name]()
    score = test_instance.score()
    return score.keys()


# not working yet
# "faithfulness": tests.Faithfulness,
# "roar_faithfulness": tests.ROARFaithfulness,
# "roar_monotonicity": tests.ROARMonotonicity,
# "monotonicity": tests.Monotonicity,
# "roar": tests.Roar,
# "shapley": tests.Shapley,
# "shapley_corr": tests.ShapleyCorr,
# "infidelity": tests.Infidelity
