import random
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))


def random_subset(s):
    out = []
    for el in s:
        # random coin flip
        if random.randint(0, 1) == 0:
            out.append(el)
    return tuple(out)