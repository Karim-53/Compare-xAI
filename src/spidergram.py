import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assuming your data is in a pandas DataFrame df
data = {
    'xAI Method': ['Random', 'Permutation', 'Permutation_Partition', 'Partition', 'Tree_Shap_Approximation', 'Exact_Shapley_Values', 'Tree_Shap', 'Saabas', 'Kernel_Shap', 'Sage', 'Lime', 'Maple', 'Joint_Shapley'],
    'Fidelity': [48, 73, 73, 73, 50, 73, 60, 100, 100, 66, 82, 33, 42],
    'Fragility': [6, 56, 56, 56, 100, 56, 100, 100, 56, 100, 100, 56, 48],
    'Stability': [11, 99, 99, 99, 74, 99, 99, 73, 99, 93, 99, 100, 98],
    'Stress': [50, 55, 55, 50, np.nan, np.nan, np.nan, np.nan, 100, 100, 100, 55, np.nan]
    # 'Time per Test': [0.0075, 9, 12, 7, 0.0057, 1906, 0.0004, 0.0012, 121, 18, 259, 56, 1947],
    # 'Completed Tests': [11, 11, 11, 11, 8, 11, 8, 8, 10, 9, 10, 11, 11]
}
df = pd.DataFrame(data).fillna(0)

# Number of variables we're plotting.
num_vars = len(df.columns) - 1

# Split the circle into even parts and save the angles
# so we know where to put each axis.
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is a circle, so we need to "complete the loop"
# and append the start to the end.
angles += angles[:1]

# ax = plt.subplot(polar=True)
fig, ax = plt.subplots(subplot_kw=dict(polar=True))

# Helper function to plot each xAI method on the spider chart.
def add_to_spider_chart(ax, angles, data_row, color):
    values = data_row.drop('xAI Method').tolist()
    values += values[:1]
    assert len(angles) == len(values), f'{angles} {values}'
    ax.plot(angles, values, color=color, linewidth=1, label=data_row['xAI Method'])
    # ax.fill(angles, values, color=color, alpha=0.25)

# Loop through each row in the DataFrame and add it to the spider chart.
for i, row in df.iterrows():
    add_to_spider_chart(ax, angles, row, None)

# Polar axes have one label by default, we want to add more.
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw axis lines for each angle and label.
labels = df.columns.drop('xAI Method').tolist()
# assert len(angles) == len(labels), f'{np.degrees(angles)} {labels}'
ax.set_thetagrids(np.degrees(angles)[:-1], labels)

# Go through labels and adjust alignment based on where
# it is in the circle.
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')

plt.yticks([25,50,75,100], ["25%","50%","75%","100%"]) # , color="grey", size=7
ax.set_ylim(0, 100)
ax.set_rlabel_position(180 / num_vars)

# ax.set_title('xAI Methods Spider Chart', size=20, color='blue', y=1.1)
# ax.legend(loc='upper right', bbox_to_anchor)
from math import pi
#
# # Draw ylabels
# ax.set_rlabel_position(0)
# plt.ylim(0,100)
#
# # Plot each individual = each line of the data
# for i, row in df.iterrows():
#     values = df.loc[i].drop('xAI Method').values.flatten().tolist()
#     values += values[:1]
#     ax.plot(angles, values, linewidth=1, linestyle='solid', label=row['xAI Method'])
#     ax.fill(angles, values, 'b', alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(2.1, 1.))
plt.tight_layout(w_pad=5)


# plt.title('Box plot of test status per xAI algorithm.')

plt.show()
# plt.savefig('boxplot.pdf', bbox_inches="tight")
# plt.savefig('boxplot.png', bbox_inches="tight")
