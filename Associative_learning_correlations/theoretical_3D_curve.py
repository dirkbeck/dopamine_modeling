import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def softmax(q_values, beta):
    q_values = np.asarray(q_values, dtype=float)
    max_q = np.max(beta * q_values)
    exp_q = np.exp(beta * q_values - max_q)
    sum_exp_q = np.sum(exp_q)
    if sum_exp_q <= 1e-12 or np.any(np.isnan(exp_q)):
        return np.ones_like(q_values) / len(q_values)
    policy = exp_q / sum_exp_q
    return np.ones_like(q_values) / len(q_values) if np.any(np.isnan(policy)) else policy

def calculate_entropy(policy):
    policy = np.clip(np.asarray(policy), 1e-12, 1.0)
    policy = policy / np.sum(policy)
    nonzero_policy = policy[policy > 1e-12]
    if len(nonzero_policy) == 0: return 0.0
    logs = np.log(nonzero_policy)
    if not np.all(np.isfinite(logs)):
        finite_logs = logs[np.isfinite(logs)]
        finite_policy = nonzero_policy[np.isfinite(logs)]
        return 0.0 if len(finite_policy) == 0 else -np.sum(finite_policy * finite_logs)
    return -np.sum(nonzero_policy * logs)

def calculate_entropy_change(old_policy, new_policy):
    h_new, h_old = calculate_entropy(new_policy), calculate_entropy(old_policy)
    return np.nan if not np.isfinite(h_new) or not np.isfinite(h_old) else h_new - h_old

def get_uncertainty_reduction(delta_q_total, q_initial, pi_old):
    q_values_new = q_initial.copy()
    q_values_new[0] = q_initial[0] + delta_q_total
    pi_new_exact = softmax(q_values_new, beta)
    entropy_change = calculate_entropy_change(pi_old, pi_new_exact)
    return np.nan if np.isnan(entropy_change) else -entropy_change

def add_arrow_triangle(ax, end_x, end_y, end_z, dir_x, dir_y, dir_z, color, scale=1.0):
    if abs(dir_z) < 0.9:
        perp_x, perp_y, perp_z = -dir_y, dir_x, 0
    else:
        perp_x, perp_y, perp_z = 0, -dir_z, dir_y

    perp_mag = np.sqrt(perp_x**2 + perp_y**2 + perp_z**2)
    perp_x, perp_y, perp_z = perp_x/perp_mag, perp_y/perp_mag, perp_z/perp_mag

    perp2_x = dir_y * perp_z - dir_z * perp_y
    perp2_y = dir_z * perp_x - dir_x * perp_z
    perp2_z = dir_x * perp_y - dir_y * perp_x

    arrow_size = scale * 0.3
    back_dist = arrow_size * 1
    side_dist = arrow_size * .4

    p1 = [end_x, end_y, end_z]
    p2 = [end_x - dir_x * back_dist + perp_x * side_dist,
          end_y - dir_y * back_dist + perp_y * side_dist,
          end_z - dir_z * back_dist + perp_z * side_dist]
    p3 = [end_x - dir_x * back_dist - perp_x * side_dist,
          end_y - dir_y * back_dist - perp_y * side_dist,
          end_z - dir_z * back_dist - perp_z * side_dist]

    triangle = np.array([p1, p2, p3])
    return ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color=color, alpha=1.0, shade=True)


plt.rcParams['font.family'] = 'Arial'
alpha, beta = 0.1, 2
delta_reward_pe_range = np.arange(-50, 50, 1)
delta_novelty_signal_range = np.arange(-2, 30, 1)
q_values_initial = np.array([1.0, 0.0])
pi_old_exact = softmax(q_values_initial, beta)

delta_q_rpe_only = alpha * delta_reward_pe_range
delta_reward_rpe_line = delta_q_rpe_only
delta_novelty_rpe_line = np.zeros_like(delta_reward_pe_range)
uncertainty_rpe_line = np.array([get_uncertainty_reduction(dq, q_values_initial, pi_old_exact) for dq in delta_q_rpe_only])

delta_q_novelty_only = alpha * delta_novelty_signal_range
delta_reward_novelty_line = np.zeros_like(delta_novelty_signal_range)
delta_novelty_novelty_line = delta_q_novelty_only
uncertainty_novelty_line = np.array([get_uncertainty_reduction(dq, q_values_initial, pi_old_exact) for dq in delta_q_novelty_only])

delta_R_grid, delta_N_grid = np.meshgrid(delta_reward_pe_range, delta_novelty_signal_range)
delta_reward_surface = alpha * delta_R_grid
delta_novelty_surface = alpha * delta_N_grid
delta_q_total_surface = alpha * (delta_R_grid + delta_N_grid)

uncertainty_surface = np.zeros_like(delta_q_total_surface)
for i in range(delta_q_total_surface.shape[0]):
    for j in range(delta_q_total_surface.shape[1]):
        uncertainty_surface[i, j] = get_uncertainty_reduction(delta_q_total_surface[i, j], q_values_initial, pi_old_exact)

fig = plt.figure(figsize=(4, 3), dpi=300)
ax = fig.add_subplot(111, projection='3d')

uncertainty_pos = uncertainty_surface.copy()
uncertainty_neg = uncertainty_surface.copy()
uncertainty_pos[uncertainty_pos < 0] = np.nan
uncertainty_neg[uncertainty_neg > 0] = np.nan

surf_pos = ax.plot_surface(delta_reward_surface, delta_novelty_surface, uncertainty_pos,
                          color='#E41A1C', alpha=0.35, rcount=100, ccount=100)
surf_neg = ax.plot_surface(delta_reward_surface, delta_novelty_surface, uncertainty_neg,
                          color='#0072B2', alpha=0.35, rcount=100, ccount=100)

x_rpe, y_rpe, z_rpe = delta_reward_rpe_line, delta_novelty_rpe_line, uncertainty_rpe_line
x_nov, y_nov, z_nov = delta_reward_novelty_line, delta_novelty_novelty_line, uncertainty_novelty_line

end_idx, start_idx = -1, -10
dir_x_rpe = x_rpe[end_idx] - x_rpe[start_idx]
dir_y_rpe = y_rpe[end_idx] - y_rpe[start_idx]
dir_z_rpe = z_rpe[end_idx] - z_rpe[start_idx]
mag_rpe = np.sqrt(dir_x_rpe**2 + dir_y_rpe**2 + dir_z_rpe**2)
if mag_rpe > 0:
    dir_x_rpe, dir_y_rpe, dir_z_rpe = dir_x_rpe/mag_rpe, dir_y_rpe/mag_rpe, dir_z_rpe/mag_rpe

dir_x_nov = x_nov[end_idx] - x_nov[start_idx]
dir_y_nov = y_nov[end_idx] - y_nov[start_idx]
dir_z_nov = z_nov[end_idx] - z_nov[start_idx]
mag_nov = np.sqrt(dir_x_nov**2 + dir_y_nov**2 + dir_z_nov**2)
if mag_nov > 0:
    dir_x_nov, dir_y_nov, dir_z_nov = dir_x_nov/mag_nov, dir_y_nov/mag_nov, dir_z_nov/mag_nov

ax.plot(x_rpe, y_rpe, z_rpe, color='#FF7F00', linestyle='--', linewidth=2.5, label='RPE-only Effect')
add_arrow_triangle(ax, x_rpe[end_idx], y_rpe[end_idx], z_rpe[end_idx],
                  dir_x_rpe, dir_y_rpe, dir_z_rpe, '#FF7F00', scale=2.0)

ax.plot(x_nov, y_nov, z_nov, color='#4DAF4A', linestyle=':', linewidth=2.5, label='Novelty-only Effect')
add_arrow_triangle(ax, x_nov[end_idx], y_nov[end_idx], z_nov[end_idx],
                  dir_x_nov, dir_y_nov, dir_z_nov, '#4DAF4A', scale=2.0)

ax.scatter(0, 0, 0, color='black', s=80, marker='o', depthshade=False)

ax.set_xlabel('Δ Utility', fontsize=12, labelpad=15)
ax.set_ylabel('Δ Novelty', fontsize=12, labelpad=15)
ax.set_zlabel('Δ Info gain', fontsize=12, labelpad=15)
ax.view_init(elev=30, azim=230)

pos_proxy = mpl.lines.Line2D([0], [0], linestyle="none", c='#E41A1C', marker='s')
neg_proxy = mpl.lines.Line2D([0], [0], linestyle="none", c='#0072B2', marker='s')
rpe_proxy = mpl.lines.Line2D([0], [0], linestyle='--', color='#FF7F00', lw=2.5)
nov_proxy = mpl.lines.Line2D([0], [0], linestyle=':', color='#4DAF4A', lw=2.5)
origin_proxy = mpl.lines.Line2D([0], [0], linestyle="none", marker='o', color='black', markersize=8)

legend = ax.legend([pos_proxy, neg_proxy, rpe_proxy, nov_proxy, origin_proxy],
          ['Positive Uncertainty Reduction', 'Negative Uncertainty Reduction',
           'Utility axis', 'Novelty axis', 'Origin'],
          loc='upper left', fontsize=8)
legend.get_frame().set_linewidth(0.5)
legend.get_frame().set_edgecolor('black')

plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('utility_novelty_axes.pdf', format='pdf', bbox_inches='tight')
plt.show()