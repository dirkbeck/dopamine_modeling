import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

np.random.seed(0)


def softmax(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: return np.array([])

    if not np.all(np.isfinite(x)):
        finite_x = x[np.isfinite(x)]
        if len(finite_x) == 0: return np.ones_like(x) / x.size
        max_finite = np.max(finite_x)
        large_neg = max_finite - 700
        x = np.nan_to_num(x, nan=large_neg, posinf=max_finite, neginf=large_neg)

    exp_x = np.exp(x - np.max(x))
    sum_exp_x = exp_x.sum()
    if sum_exp_x == 0 or not np.isfinite(sum_exp_x):
        return np.ones_like(x) / x.size
    return exp_x / sum_exp_x


def entropy(probs):
    probs = np.asarray(probs)
    probs = np.maximum(0, probs)
    probs_sum = probs.sum()

    if not np.isclose(probs_sum, 1.0):
        if probs_sum > 1e-9:
            probs = probs / probs_sum
        else:
            return 0

    nonzero_probs = probs[probs > 1e-12]
    if len(nonzero_probs) == 0: return 0
    logs = np.log(nonzero_probs)
    if not np.all(np.isfinite(logs)): return np.nan
    return -np.sum(nonzero_probs * logs)


def d1_only_probabilities(Q, beta_d1):
    return softmax(beta_d1 * np.array(Q, dtype=float))


def d2_only_probabilities(B, beta_d2):
    return softmax(beta_d2 * np.array(B, dtype=float))


def d1d2_probabilities(Q, B, beta_d1, beta_d2):
    combined_value = beta_d1 * np.array(Q, dtype=float) + beta_d2 * np.array(B, dtype=float)
    return softmax(combined_value)


# data
Q = [50, 49.95, 49.9, 40, 35, 30, 25, 20, 15, 10, 5, 0.5]
B = [-0.5, -0.5, -0.5, 0, 0, 0, 0, 0, 0, 0.5, 0.5, 0.5]

beta1_range_d1_only = np.linspace(0.001, 10, 200)
beta1_range_d1d2 = np.linspace(0.001, 100, 200)
beta2_range_d2_only = np.linspace(-10, 10, 200)
beta2_range_d1d2 = np.linspace(-10, 10, 200)

# calculate entropies
d1_only_entropies = [entropy(d1_only_probabilities(Q, b1)) for b1 in beta1_range_d1_only]
d2_only_entropies = [entropy(d2_only_probabilities(B, b2)) for b2 in beta2_range_d2_only]

entropy_map = np.zeros((len(beta1_range_d1d2), len(beta2_range_d1d2)))
for i, beta1 in enumerate(beta1_range_d1d2):
    for j, beta2 in enumerate(beta2_range_d1d2):
        entropy_map[i, j] = entropy(d1d2_probabilities(Q, B, beta1, beta2))

entropy_map = np.nan_to_num(entropy_map, nan=np.log(len(Q)))
max_theoretical_entropy = np.log(len(Q))

# get min/max values for plotting
d1_min = min(d1_only_entropies)
d1_max = max(d1_only_entropies)
d2_min = min(d2_only_entropies)
d2_max = max(d2_only_entropies)
d1d2_min = np.min(entropy_map)
d1d2_max = np.max(entropy_map)

target_range = np.linspace(0, max_theoretical_entropy * 1.01, 100)
all_targets = np.unique(target_range[target_range >= 0])

# find optimal parameters for d1d2
optimal_params_d1d2 = []
for target in all_targets:
    error_map = np.abs(entropy_map - target)
    min_idx = np.unravel_index(np.argmin(error_map), error_map.shape)
    achieved = entropy_map[min_idx]
    optimal_params_d1d2.append((beta1_range_d1d2[min_idx[0]], beta2_range_d1d2[min_idx[1]], achieved, target))
optimal_params_d1d2 = np.array(optimal_params_d1d2)

# find closest entropies for d1 and d2 only
closest_d1_entropies = []
d1_errors = []
d1_arr = np.array(d1_only_entropies)
for target in all_targets:
    idx = np.argmin(np.abs(d1_arr - target))
    closest = d1_arr[idx]
    closest_d1_entropies.append(closest)
    d1_errors.append(abs(closest - target))

closest_d2_entropies = []
d2_errors = []
d2_arr = np.array(d2_only_entropies)
for target in all_targets:
    idx = np.argmin(np.abs(d2_arr - target))
    closest = d2_arr[idx]
    closest_d2_entropies.append(closest)
    d2_errors.append(abs(closest - target))

d1d2_errors = np.abs(optimal_params_d1d2[:, 2] - all_targets)

# plotting
fig = plt.figure(figsize=(12, 4), dpi=300)
gs = GridSpec(1, 7, figure=fig, width_ratios=[1.2, 0.05, 1.2, 0.05, 1.2, 0.05, 0.6])

panel_labels = ['A', 'B', 'C']
label_x, label_y = -0.1, 1.12
near_zero_threshold = 1e-3
combined_min = min(d1_min, d2_min)
combined_max = max(d1_max, d2_max)

# entropy control capability
ax1 = fig.add_subplot(gs[0, 0])

if d1d2_min > near_zero_threshold:
    ax1.axhspan(0, d1d2_min, color='lightgray', alpha=0.5, hatch='///')

lower_exclusive = None
if d1d2_min < combined_min - near_zero_threshold:
    lower_exclusive = ax1.axhspan(d1d2_min, combined_min, color='lightgreen', alpha=0.3)

upper_exclusive = None
if d1d2_max > combined_max:
    plot_upper_bound = min(d1d2_max, max_theoretical_entropy * 1.02)
    upper_exclusive = ax1.axhspan(combined_max, plot_upper_bound, color='lightgreen', alpha=0.3)

perfect_line = ax1.plot(all_targets, all_targets, 'k--', alpha=0.7, linewidth=2.0)[0]
d1_line = ax1.plot(all_targets, closest_d1_entropies, 'b-', linewidth=2.0)[0]
d2_line = ax1.plot(all_targets, closest_d2_entropies, 'r-', linewidth=2.0)[0]
d1d2_line = ax1.plot(all_targets, optimal_params_d1d2[:, 2], 'g-', linewidth=3.0)[0]
max_line = ax1.axhline(y=max_theoretical_entropy, color='purple', linestyle=':', alpha=0.8, linewidth=1.5)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlabel('Target entropy', fontsize=14)
ax1.set_ylabel('Achieved entropy', fontsize=14)
ax1.set_xlim(-0.05, max_theoretical_entropy * 1.05)
ax1.set_ylim(-0.05, max_theoretical_entropy * 1.05)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.text(label_x, label_y, panel_labels[0], transform=ax1.transAxes,
         fontsize=18, fontweight='bold', va='top', ha='right')

# error in achieving target entropy
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(all_targets, d1_errors, 'b-', linewidth=2.0, label='D1 Only')
ax2.plot(all_targets, d2_errors, 'r-', linewidth=2.0, label='D2 Only')
ax2.plot(all_targets, d1d2_errors, 'g-', linewidth=3.0, label='D1+D2')
ax2.set_yscale('log')

ax2.axvline(x=d1_min, color='blue', linestyle=':', alpha=0.6)
ax2.axvline(x=d1_max, color='blue', linestyle=':', alpha=0.6)
ax2.axvline(x=d2_min, color='red', linestyle=':', alpha=0.6)
ax2.axvline(x=d2_max, color='red', linestyle=':', alpha=0.6)
ax2.axvline(x=d1d2_min, color='green', linestyle='--', alpha=0.6)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('Target entropy', fontsize=14)
ax2.set_ylabel('Absolute error (log)', fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.text(label_x, label_y, panel_labels[1], transform=ax2.transAxes,
         fontsize=18, fontweight='bold', va='top', ha='right')

# optimal D1/D2 parameters
ax3 = fig.add_subplot(gs[0, 4])
sorted_indices = np.argsort(optimal_params_d1d2[:, 3])
target_entropies = optimal_params_d1d2[sorted_indices, 3]
optimal_beta1 = optimal_params_d1d2[sorted_indices, 0]
optimal_beta2 = optimal_params_d1d2[sorted_indices, 1]

norm = plt.Normalize(min(target_entropies), max(target_entropies))
cmap = plt.cm.viridis
points = np.array([optimal_beta1, optimal_beta2]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(target_entropies)
lc.set_linewidth(2.5)
ax3.add_collection(lc)

ax3.set_xlim(min(optimal_beta1) * 0.9, max(optimal_beta1) * 1.05)
ax3.set_ylim(min(optimal_beta2) * 1.1 if min(optimal_beta2) < 0 else min(optimal_beta2) * 0.9,
             max(optimal_beta2) * 1.05)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.set_xlabel('D1 sensitivity', fontsize=14)
ax3.set_ylabel('D2 sensitivity', fontsize=14)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.text(label_x, label_y, panel_labels[2], transform=ax3.transAxes,
         fontsize=18, fontweight='bold', va='top', ha='right')

# colorbar
cax = fig.add_subplot(gs[0, 6])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label('Target entropy', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# legend
legend_items = [d1_line, d2_line, d1d2_line, perfect_line, max_line]
legend_labels = ['D1 Only', 'D2 Only', 'D1+D2', 'Target=Achieved', f'Max ({max_theoretical_entropy:.2f})']

if lower_exclusive or upper_exclusive:
    exclusive_patch = mpatches.Patch(color='lightgreen', alpha=0.3)
    legend_items.append(exclusive_patch)
    legend_labels.append('D1+D2 only')

if d1d2_min > near_zero_threshold:
    hatched_patch_legend = mpatches.Patch(color='lightgray', alpha=0.5, hatch='///')
    legend_items.append(hatched_patch_legend)
    legend_labels.append('Unreachable')

fig.legend(legend_items, legend_labels, loc='lower center', ncol=6, fontsize=10,
           bbox_to_anchor=(0.5, -0.05), frameon=False)

plt.tight_layout(pad=1.5, rect=[0.01, 0.05, 0.99, 0.95])
plt.savefig('entropy_analysis_3panel.pdf', dpi=600, bbox_inches='tight')
plt.show()