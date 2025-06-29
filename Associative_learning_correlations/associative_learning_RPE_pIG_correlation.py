import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

np.random.seed(110)

# params (selected arbitrarily)
num_trials = 200
learning_rate = 0.15
reward_magnitude = 1.0
beta = 3.0
w_V = -1.0
epsilon = 1e-9
plot_len_early = 20
plot_len_late = 20
gap_size = 5

def softmax(q_values, gamma):
    q_values = np.asarray(q_values, dtype=float)
    scaled_q = gamma * q_values
    max_q = np.max(scaled_q)
    stable_q = scaled_q - max_q
    exp_q = np.exp(np.clip(stable_q, -700, 700))
    return exp_q / np.sum(exp_q)

def shannon_entropy(probs):
    probs = probs[probs > epsilon]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log(probs))

def run_simulation(reward_prob=1.0):
    q_cue = 0.0
    q_baseline = 0.0
    H_max = np.log(2)

    rpe_history = []
    info_gain_history = []
    delta_info_gain_history = []

    for t in range(num_trials):
        q_array = np.array([q_cue, q_baseline])
        probs_before = softmax(q_array, beta)
        entropy_before = shannon_entropy(probs_before)

        info_gain = max(0, H_max - entropy_before)
        info_gain_history.append(info_gain)

        # get reward
        reward = reward_magnitude if np.random.rand() < reward_prob else 0.0
        rpe = reward - q_cue
        rpe_history.append(rpe)

        # update Q
        q_cue = np.clip(q_cue + learning_rate * rpe, -0.2, 1.2)

        q_array_after = np.array([q_cue, q_baseline])
        probs_after = softmax(q_array_after, beta)
        entropy_after = shannon_entropy(probs_after)

        delta_info_gain = entropy_before - entropy_after
        delta_info_gain_history.append(delta_info_gain)

    return {
        'rpe': np.array(rpe_history),
        'info_gain': np.array(info_gain_history),
        'delta_info_gain': np.array(delta_info_gain_history)
    }

# run both conditions
det_results = run_simulation(reward_prob=1.0)
prob_results = run_simulation(reward_prob=0.8)

# prep data
x_early = np.arange(plot_len_early)
x_late = np.arange(plot_len_early + gap_size, plot_len_early + gap_size + plot_len_late)

det_rpe_early = det_results['rpe'][:plot_len_early]
det_rpe_late = det_results['rpe'][-plot_len_late:]
prob_rpe_early = prob_results['rpe'][:plot_len_early]
prob_rpe_late = prob_results['rpe'][-plot_len_late:]

det_delta_ig_early = det_results['delta_info_gain'][:plot_len_early]
det_delta_ig_late = det_results['delta_info_gain'][-plot_len_late:]
prob_delta_ig_early = prob_results['delta_info_gain'][:plot_len_early]
prob_delta_ig_late = prob_results['delta_info_gain'][-plot_len_late:]

det_ig_early = det_results['info_gain'][:plot_len_early]
det_ig_late = det_results['info_gain'][-plot_len_late:]
prob_ig_early = prob_results['info_gain'][:plot_len_early]
prob_ig_late = prob_results['info_gain'][-plot_len_late:]


plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})
fig, axes = plt.subplots(2, 4, figsize=(7.09, 2.0))
fig.subplots_adjust(wspace=0.1, hspace=0.15, left=0.06, right=0.98, top=0.95, bottom=0.15)

# plot bars for RPE - deterministic
colors_early = ['forestgreen' if r >= 0 else 'firebrick' for r in det_rpe_early]
colors_late = ['forestgreen' if r >= 0 else 'firebrick' for r in det_rpe_late]
axes[0, 0].bar(x_early, det_rpe_early, color=colors_early, width=0.8)
axes[0, 0].bar(x_late, det_rpe_late, color=colors_late, width=0.8)
axes[0, 0].set_ylim((-1, 1))
axes[0, 0].axhline(y=0, color='black', lw=0.5)
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)
axes[0, 0].set_xticks([0, 40])
axes[0, 0].set_yticks([-1, 0, 1])
axes[0, 0].set_ylabel("RPE", fontsize=6)
green_patch = mpatches.Patch(color='forestgreen', label='RPE ≥ 0')
red_patch = mpatches.Patch(color='firebrick', label='RPE < 0')
axes[0, 0].legend(handles=[green_patch, red_patch], loc='upper right', frameon=False, fontsize=6)

# delta info gain - deterministic
if len(det_delta_ig_early) > 0:
    axes[0, 1].plot(x_early, det_delta_ig_early, c='red', lw=1)
if len(det_delta_ig_late) > 0:
    axes[0, 1].plot(x_late, det_delta_ig_late, c='red', lw=1)
axes[0, 1].set_ylim((-0.1, 0.1))
axes[0, 1].axhline(0, color='grey', alpha=0.5, lw=0.5)
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)
axes[0, 1].set_xticks([0, 40])
axes[0, 1].set_yticks([-0.1, 0, 0.1])
axes[0, 1].set_ylabel("Δ Policy-IG\n(nats)", fontsize=6)

# info gain filled - deterministic
if len(det_ig_early) > 0:
    axes[0, 2].plot(x_early, det_ig_early, c='magenta', lw=1)
    axes[0, 2].fill_between(x_early, 0, det_ig_early, color='plum', alpha=0.5)
if len(det_ig_late) > 0:
    axes[0, 2].plot(x_late, det_ig_late, c='magenta', lw=1)
    axes[0, 2].fill_between(x_late, 0, det_ig_late, color='plum', alpha=0.5)
axes[0, 2].set_ylim((0, 0.7))
axes[0, 2].axhline(0, color='grey', alpha=0.5, lw=0.5)
axes[0, 2].spines['top'].set_visible(False)
axes[0, 2].spines['right'].set_visible(False)
axes[0, 2].set_xticks([0, 40])
axes[0, 2].set_yticks([0, 0, 0.7])
axes[0, 2].set_ylabel("Policy-IG\n(nats)", fontsize=6)

# scatter plot - deterministic
mask = np.isfinite(det_results['rpe']) & np.isfinite(det_results['delta_info_gain'])
x_clean = det_results['rpe'][mask]
y_clean = det_results['delta_info_gain'][mask]
if len(x_clean) >= 2:
    axes[0, 3].scatter(x_clean, y_clean, alpha=0.3, color='red', s=20, edgecolors='none')
    slope, intercept, r_value, p_value, _ = linregress(x_clean, y_clean)
    if abs(r_value) > 0.1:
        line_x = np.array([np.min(x_clean), np.max(x_clean)])
        line_y = slope * line_x + intercept
        axes[0, 3].plot(line_x, line_y, c='k', lw=1.2, alpha=0.8)
    axes[0, 3].text(0.05, 0.95, f'R={r_value:.2f}', transform=axes[0, 3].transAxes, fontsize=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'))
else:
    axes[0, 3].text(0.5, 0.5, 'No Data', transform=axes[0, 3].transAxes, ha='center', va='center')
axes[0, 3].set_xlim((-1, 1))
axes[0, 3].set_ylim((-0.1, 0.1))
axes[0, 3].axhline(0, c='grey', alpha=0.5, lw=0.5)
axes[0, 3].axvline(0, c='grey', alpha=0.5, lw=0.5)
axes[0, 3].spines['top'].set_visible(False)
axes[0, 3].spines['right'].set_visible(False)
axes[0, 3].set_xticks([-1, 0, 1])
axes[0, 3].set_yticks([-0.1, 0, 0.1])
axes[0, 3].set_ylabel("Δ Policy-IG\n(nats)", fontsize=6)

# second row - probabilistic
colors_early = ['forestgreen' if r >= 0 else 'firebrick' for r in prob_rpe_early]
colors_late = ['forestgreen' if r >= 0 else 'firebrick' for r in prob_rpe_late]
axes[1, 0].bar(x_early, prob_rpe_early, color=colors_early, width=0.8)
axes[1, 0].bar(x_late, prob_rpe_late, color=colors_late, width=0.8)
axes[1, 0].set_ylim((-1, 1))
axes[1, 0].axhline(y=0, color='black', lw=0.5)
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)
axes[1, 0].set_xticks([0, 40])
axes[1, 0].set_yticks([-1, 0, 1])
axes[1, 0].set_ylabel("RPE", fontsize=6)

if len(prob_delta_ig_early) > 0:
    axes[1, 1].plot(x_early, prob_delta_ig_early, c='red', lw=1)
if len(prob_delta_ig_late) > 0:
    axes[1, 1].plot(x_late, prob_delta_ig_late, c='red', lw=1)
axes[1, 1].set_ylim((-0.1, 0.1))
axes[1, 1].axhline(0, color='grey', alpha=0.5, lw=0.5)
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)
axes[1, 1].set_xticks([0, 40])
axes[1, 1].set_yticks([-0.1, 0, 0.1])
axes[1, 1].set_ylabel("Δ Policy-IG\n(nats)", fontsize=6)

if len(prob_ig_early) > 0:
    axes[1, 2].plot(x_early, prob_ig_early, c='magenta', lw=1)
    axes[1, 2].fill_between(x_early, 0, prob_ig_early, color='plum', alpha=0.5)
if len(prob_ig_late) > 0:
    axes[1, 2].plot(x_late, prob_ig_late, c='magenta', lw=1)
    axes[1, 2].fill_between(x_late, 0, prob_ig_late, color='plum', alpha=0.5)
axes[1, 2].set_ylim((0, 0.7))
axes[1, 2].axhline(0, color='grey', alpha=0.5, lw=0.5)
axes[1, 2].spines['top'].set_visible(False)
axes[1, 2].spines['right'].set_visible(False)
axes[1, 2].set_xticks([0, 40])
axes[1, 2].set_yticks([0, 0, 0.7])
axes[1, 2].set_ylabel("Policy-IG\n(nats)", fontsize=6)

# scatter for prob condition
mask = np.isfinite(prob_results['rpe']) & np.isfinite(prob_results['delta_info_gain'])
x_clean = prob_results['rpe'][mask]
y_clean = prob_results['delta_info_gain'][mask]
if len(x_clean) >= 2:
    axes[1, 3].scatter(x_clean, y_clean, alpha=0.3, color='red', s=20, edgecolors='none')
    slope, intercept, r_value, p_value, _ = linregress(x_clean, y_clean)
    if abs(r_value) > 0.1:
        line_x = np.array([np.min(x_clean), np.max(x_clean)])
        line_y = slope * line_x + intercept
        axes[1, 3].plot(line_x, line_y, c='k', lw=1.2, alpha=0.8)
    axes[1, 3].text(0.05, 0.95, f'R={r_value:.2f}', transform=axes[1, 3].transAxes, fontsize=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.5, ec='none'))
else:
    axes[1, 3].text(0.5, 0.5, 'No Data', transform=axes[1, 3].transAxes, ha='center', va='center')
axes[1, 3].set_xlim((-1, 1))
axes[1, 3].set_ylim((-0.1, 0.1))
axes[1, 3].axhline(0, c='grey', alpha=0.5, lw=0.5)
axes[1, 3].axvline(0, c='grey', alpha=0.5, lw=0.5)
axes[1, 3].spines['top'].set_visible(False)
axes[1, 3].spines['right'].set_visible(False)
axes[1, 3].set_xticks([-1, 0, 1])
axes[1, 3].set_yticks([-0.1, 0, 0.1])
axes[1, 3].set_ylabel("Δ Policy-IG\n(nats)", fontsize=6)
axes[1, 3].set_xlabel("RPE", fontsize=6)


axes[1, 0].set_xlabel("Episode", fontsize=6)
axes[1, 1].set_xlabel("Episode", fontsize=6)
axes[1, 2].set_xlabel("Episode", fontsize=6)

for i in range(2):
    for j in range(4):
        axes[i, j].tick_params(labelsize=6, pad=1)
        if i == 0:
            axes[i, j].tick_params(labelbottom=False)

plt.tight_layout()
plt.savefig('rpe_pIG_correlations_associative_learning.pdf', dpi=300, bbox_inches='tight', pad_inches=0.02,
            transparent=True)
plt.show()