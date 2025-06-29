import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy.optimize import brentq

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.titlesize': 15
})

p_range = np.linspace(0.01, 1.0, 50)
y_range = np.linspace(-3, 3, 50)
P_mesh, Y_mesh = np.meshgrid(p_range, y_range)

beta_initial = 1
target_entropy = 0.1
max_entropy = np.log(2)


def calculate_p_choice(y_diff, beta):
    exp_arg = np.clip(beta * (-y_diff), -700, 700)
    return np.clip(1.0 / (1.0 + np.exp(exp_arg)), 1e-9, 1.0 - 1e-9)


def calculate_pi_u(p_avail, p_choice):
    return p_avail * p_choice


def calculate_entropy(p_choice):
    p_c = np.clip(p_choice, 1e-9, 1.0 - 1e-9)
    p_not_c = 1.0 - p_c
    return -(p_c * np.log(p_c) + p_not_c * np.log(p_not_c))


def entropy_solver(p, target_h):
    return calculate_entropy(p) - target_h


def find_p_choice_for_entropy(target_h, initial_p):
    target_h = np.clip(target_h, 1e-9, max_entropy - 1e-9)

    if np.isclose(target_h, 0):
        return 0.0 if initial_p < 0.5 else 1.0
    if np.isclose(target_h, max_entropy):
        return 0.5

    try:
        p_lower = brentq(entropy_solver, 1e-9, 0.5 - 1e-9, args=(target_h,))
    except:
        p_lower = np.clip(np.sqrt(target_h / 2), 0, 0.49)

    p_upper = 1.0 - p_lower
    return p_upper if initial_p >= 0.5 else p_lower


# calculate initial policy surface
p_choice_initial = calculate_p_choice(Y_mesh, beta_initial)
pi_u_initial = calculate_pi_u(P_mesh, p_choice_initial)

# create plot
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(P_mesh, Y_mesh, pi_u_initial, cmap='RdYlGn',
                       edgecolor='none', alpha=0.6)

ax.set_xlabel('Action availability', labelpad=18, fontsize=12)
ax.set_ylabel('Value difference (uncertain - certain)', labelpad=18, fontsize=12)
ax.set_zlabel('Choice prob.', labelpad=12, fontsize=12)

cbar = fig.colorbar(surf, shrink=0.5, aspect=15, pad=0.02, location='left')
cbar.set_label('Choice prob.', fontsize=10)

# add vector field
p_grid = np.linspace(0.15, 0.95, 4)
y_grid = np.linspace(-2.8, 2.8, 4)

for p0 in p_grid:
    for y0 in y_grid:
        p_choice_init = calculate_p_choice(y0, beta_initial)
        pi_u_init = calculate_pi_u(p0, p_choice_init)

        p_choice_new = find_p_choice_for_entropy(target_entropy, p_choice_init)
        pi_u_new = calculate_pi_u(p0, p_choice_new)

        w = 3 * (pi_u_new - pi_u_init)

        if w > 0.005:
            color = "green"
        elif w < -0.005:
            color = "red"
        else:
            color = "blue"

        if abs(w) > 0.001:
            ax.quiver(p0, y0, pi_u_init, 0, 0, w, color=color,
                      linewidth=1.5, arrow_length_ratio=0.3, normalize=False)

ax.view_init(elev=28, azim=-55)
ax.set_zlim(-0.05, P_mesh.max() + 0.05)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='green', marker='^', linestyle='None', markersize=7,
           label='Info gain increases choice prob.'),
    Line2D([0], [0], color='red', marker='v', linestyle='None', markersize=7,
           label='Info gain decreases choice prob.'),
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99),
          fontsize=9)

plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('decision_making_manifold.pdf', bbox_inches='tight')
plt.savefig('decision_making_manifold.svg', bbox_inches='tight')
plt.show()