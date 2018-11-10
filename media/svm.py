# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

scores =  [0.814179104477612, 0.7985074626865671, 0.8022388059701493, 0.7932835820895522]
alg = ('RBF', 'Polynomial', 'Sigmoid', 'Linear')

N = 4
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

matplotlib.rcParams['legend.handleheight'] = 2
matplotlib.rcParams['axes.axisbelow'] = True

fig, ax = plt.subplots()
ax.grid(True)

rects = ax.bar(ind, scores, width, color='#00FFFF', edgecolor='black', hatch="/")

# add some text for labels, title and axes ticks
ax.set_title('SVM Kernel Accuracy - MIT Indoor 67')
ax.set_ylabel('Accuracy')
ax.set_xticks(ind)
ax.set_xticklabels(alg)

# ax.legend((rects[0]), ('Optimized', 'Not optimized', 'Another model'), prop={'size':9}, loc=2)

axes = plt.gca()
axes.set_ylim([0.6,0.9])

plt.show()
