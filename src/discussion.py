"""Numbers reported in Discussion section"""
import numpy as np
import pandas as pd

# Discussion of Level 2
df = pd.DataFrame(index=['fidelity', 'fragility', 'stability', 'simplicity', 'Stress test'])
df['baseline_random'] = [12.9, 50, 30.7, 0, 33.3]
df['exact_shapley_values'] = [100, 11.1, 84.3, 100, np.nan]
df['kernel_shap'] = [100, 11.1, 85.6, 100, 100]
df['lime'] = [89, 0, 99.7, 98.5, 40.9]
df['maple'] = [60, 11.1, 100, 0, 25.5]
df['partition'] = [100, 11.1, 84.3, 100, 21.7]
df['permutation'] = [100, 11.1, 84.3, 100, 100]
df['permutation_partition'] = [100, 11.1, 84.3, 100, 100]
df['saabas'] = [60, np.nan, 50.6, np.nan, np.nan]
df['sage'] = [66.7, 11.1, 96.9, 100, 55.7]
df['tree_shap'] = [64, np.nan, 84.3, np.nan, np.nan]
df['tree_shap_approximation'] = [100, np.nan, 49.9, np.nan, np.nan]

print(df.describe())
description = df.describe()
delta = description.loc['max'] - description.loc['min']
_mean = delta.mean()
_std = delta.std()
print('The average difference in sub scores is', round(_mean, 2), '% +-', round(_std, 2), '%')

# Level 3
df = pd.read_csv(r'C:\Inn\GitHub\Compare-xAI\data\03_experiment_output_aggregated\cross_tab.csv')
stat = pd.DataFrame(index=df.groupby('explainer').count()['test'].index)
stat['total'] = df.groupby('explainer').count()['test']
stat['partial_fail'] = df[df.score < .95].groupby('explainer').count()['test']
stat['fail'] = df[df.score < .05].groupby('explainer').count()['test']

stat['partial_fail_ratio'] = stat['partial_fail'] / stat['total'] * 100.
stat['fail_ratio'] = stat['fail'] / stat['total'] * 100.

import seaborn as sns
import matplotlib.pyplot as plt

meanprops = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markersize": "6",
    "label": "Mean",
}




fig, ax = plt.subplots(1, 2, figsize=(6, .6), sharex=True, sharey=True)
for i in range(2):
    ax[i].tick_params(axis=u'both', which=u'both',length=0)
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].set_xlim([0,95])

sns.boxplot(x=stat.fail_ratio, ax=ax[0],
            showmeans=True,
            meanprops=meanprops,
            )
# ax[0].grid(axis="x")
ax[0].set_xlabel('Failed tests [%]')

# ax[0].spines['top'].set_visible(False)
# ax[0].spines['right'].set_visible(False)

ax[0].set_yticks([])

# axis 1
sns.boxplot(x=stat.partial_fail_ratio, ax=ax[1], # width=.7,
            showmeans=True,
            meanprops=meanprops,
            )
# ax[1].grid(axis="x")
ax[1].set_xlabel('Partially failed tests [%]')

ax[1].legend(loc=(1.03, 0))

plt.tight_layout(pad=0)
# plt.title('Box plot of test status per xAI algorithm.')
# plt.show()
plt.savefig('boxplot.pdf', bbox_inches="tight")
plt.savefig('boxplot.png', bbox_inches="tight")
