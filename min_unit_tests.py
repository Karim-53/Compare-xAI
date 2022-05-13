""" Study the stability of the benchmark """
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paretoset import paretoset

n = 20  # xai
m = 20
t = 100  # iteration


def experience(n, m):
    summary_df = pd.DataFrame()
    summary_df['percentage'] = np.random.beta(4, 4, n)  # like normal distrib but cut
    summary_df['time_per_test'] = abs(np.random.randn(n))  # half gaussian
    mask1 = paretoset(summary_df[['percentage', 'time_per_test']], sense=["max", "min"])

    summary_df['percentage'] = (summary_df['percentage'] * m + np.random.beta(.6, .6, n)) / (m + 1)
    summary_df['time_per_test'] = (summary_df['time_per_test'] * m + abs(np.random.randn(n))) / (m + 1)
    mask2 = paretoset(summary_df[['percentage', 'time_per_test']], sense=["max", "min"])

    return sum(mask1 != mask2)  # / n


def smallest_acceptable_nb_tests(n, t):
    for m in range(n // 4, 4 * n, 3):
        ex = [experience(n, m) for _ in range(t)]
        print(n, m, np.average(ex), pd.Series(ex).sem())
        if np.average(ex) < 1:
            return m
    else:
        raise Exception('we need more than unit tests...')


# ex = [experience(n, m) for _ in range(t)]
# print(np.average(ex), pd.Series(ex).sem())

# x = []
# lista = []
# for m in range(n // 4, 2 * n, 5):
#     ex = [experience(n, m) for _ in range(t)]
#     print(m, np.average(ex), pd.Series(ex).sem())
#     lista.append(np.average(ex))
#     x.append(m)
# plt.plot(x, lista)
# plt.title(f'n = {n}')
# plt.show()

n_lista = list(range(10, 1000, 10))
m_lista = []
for n in n_lista:
    m_lista.append(smallest_acceptable_nb_tests(n, t))

plt.figure(figsize=(5, 5))
plt.plot(n_lista, m_lista, label='$M = f(n)$')
plt.plot(n_lista, n_lista, '--', color='gray', label='diagonal')
plt.plot(n_lista, [x * 0.5 for x in n_lista], '--', label='trend line: $M = 0.5 * n$')
plt.xlabel('$n$: the number of xAI')
plt.ylabel('$M$: the minimum number of unit tests')
plt.xlim([0, 200])
plt.ylim([0, 200])
# plt.title(f'threshold')
plt.grid()
plt.legend()
plt.show()
