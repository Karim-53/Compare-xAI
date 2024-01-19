import pandas as pd
# df = pd.DataFrame([[],
#               []],
#              columns=,
#              index = ['', ''],
#              )

import matplotlib.pyplot as plt
import numpy as np

species = ['Fidelity [%]', 'Fragility [%]', 'Stability [%]', 'Stress test [%]',]#  'Time per\nTest [Sec]', 'Portability']
penguin_means = {
    'Kernel Shap': (100, 56, 99, 100),  # , 121, 10),
    'Sage': (66, 100, 93, 100),  # , 18, 9),
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# ax.set_ylabel('Length (mm)')
ax.set_xticks(x + width, species)
ax.legend(loc='lower left', ncols=2)
# ax.set_ylim(0, 250)

plt.show()
