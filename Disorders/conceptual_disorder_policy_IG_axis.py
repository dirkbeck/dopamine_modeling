import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'figure.dpi': 300,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

t = np.linspace(0, 1, 200)
line = np.vstack((t, t, t)).T

cmap = plt.get_cmap('coolwarm')

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for i in range(len(t) - 1):
    ax.plot(line[i:i + 2, 0], line[i:i + 2, 1], line[i:i + 2, 2],
            color=cmap(t[i]), linewidth=3, alpha=0.8)

ax.set_xlabel('Learning benefit', fontsize=12)
ax.set_ylabel('Policy balance', fontsize=12)
ax.set_zlabel('Information throughput', fontsize=12)

import matplotlib as mpl
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=12)
cbar.set_label('Policy-IG Value', fontsize=11)

plt.tight_layout()
plt.savefig('policy_ig_figure_final_layout.pdf', format='pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

plt.show()