import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 10
rcParams['axes.linewidth'] = 1
rcParams['pdf.fonttype'] = 42
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['legend.frameon'] = False

model_uncertainty = np.linspace(0, 1, 100)  # Low (0) to High (1)
value_uncertainty = np.linspace(0, 1, 100)  # Low (0) to High (1)
X, Y = np.meshgrid(model_uncertainty, value_uncertainty)

delta_dopamine = X - Y  # Delta DA / Delta InfoGain ~ (Model Uncertainty) - (Value Uncertainty)

fig, ax = plt.subplots(figsize=(4.5, 4))

vmin, vmax = -1, 1
levels = np.linspace(vmin, vmax, 15)

contour = ax.contourf(X, Y, delta_dopamine, levels=levels, cmap='coolwarm',
                      vmin=vmin, vmax=vmax, extend='both')

cbar = fig.colorbar(contour, shrink=0.8, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
cbar.set_label(r'$\Delta$ Dopamine / $\Delta$ InfoGain (arb. u.)', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.contour(X, Y, delta_dopamine, levels=[0], colors='black', linewidths=1.5, linestyles='--')
ax.contour(X, Y, delta_dopamine, levels=levels, colors='white', linewidths=0.3, alpha=0.3)

ax.set_xlabel(r'Model/state uncertainty')
ax.set_ylabel(r'Value uncertainty')

markers = [(0.85, 0.1, '*', 12, 'Neutral Novelty'),
           (0.85, 0.85, 's', 8, 'New Environment'),
           (0.15, 0.85, '^', 8, 'Value Learning'),
           (0.15, 0.1, 'o', 8, 'Stable Exploitation')]

for x, y, marker, size, desc in markers:
    ax.plot(x, y, 'k'+marker, markersize=size)


legend_elements = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor='k',
                             markersize=10 if m == '*' else 8, label=d)
                   for _, _, m, _, d in markers]

ax.legend(handles=legend_elements, loc='upper center',
          bbox_to_anchor=(0.5, -0.15), ncol=2)

ax.set_xticks([0, 0.5, 1])
ax.set_yticks([0, 0.5, 1])
ax.set_xticklabels(['Low', 'Med', 'High'])
ax.set_yticklabels(['Low', 'Med', 'High'])
ax.set_aspect('equal')

plt.tight_layout()

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig("novelty_colormap.pdf", dpi=300, bbox_inches='tight', format='pdf')
plt.show()