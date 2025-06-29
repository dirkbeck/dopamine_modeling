import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.linewidth': 1.0,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

delta_L_val = 20.0
temperatures = np.logspace(2, -2.5, 400)
action_counts = {'Song': 10, 'Note': 100, 'Motor primative': 10000}
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
line_data = {}
color_idx = 0

for label, N_actions in action_counts.items():
    entropies_list = []
    probs_optimal_list = []
    max_H_for_N = np.log(N_actions) if N_actions > 1 else 0.0

    # add point for very high temperature (max entropy)
    p_opt_max_t = 1.0 / N_actions if N_actions > 0 else 0
    h_max_t = max_H_for_N
    entropies_list.append(h_max_t)
    probs_optimal_list.append(p_opt_max_t)

    for T_val in temperatures:
        if N_actions == 0:
            p_opt, entropy = 0, 0
        elif N_actions == 1:
            p_opt, entropy = 1.0, 0.0
        elif T_val < 1e-9:  # zero temperature
            p_opt, entropy = 1.0, 0.0
        else:
            # standard calculation
            exp_neg_delta_L_T = np.exp(-delta_L_val / T_val)
            p_opt = 1.0 / (1.0 + (N_actions - 1) * exp_neg_delta_L_T)

            if p_opt > 1.0 - 1e-9:
                p_other = 0.0
            else:
                p_other = p_opt * exp_neg_delta_L_T

            p_opt_clipped = np.clip(p_opt, 1e-12, 1.0)
            p_other_clipped = np.clip(p_other, 1e-12, 1.0)

            term1 = -p_opt_clipped * np.log(p_opt_clipped) if p_opt > 1e-9 else 0.0

            if N_actions > 1 and p_other > 1e-9:
                term2 = -(N_actions - 1) * p_other_clipped * np.log(p_other_clipped)
            else:
                term2 = 0.0

            entropy = term1 + term2
            entropy = np.clip(entropy, 0, max_H_for_N)

        entropies_list.append(entropy)
        probs_optimal_list.append(p_opt)

    # add low temperature point
    entropies_list.append(0.0)
    probs_optimal_list.append(1.0)

    current_entropies = np.array(entropies_list)
    current_probs_optimal = np.array(probs_optimal_list)

    sorted_indices = np.argsort(current_entropies)
    current_entropies_sorted = current_entropies[sorted_indices]
    current_probs_optimal_sorted = current_probs_optimal[sorted_indices]

    unique_entropies, unique_indices = np.unique(current_entropies_sorted, return_index=True)
    final_target_entropies = unique_entropies
    final_probs_optimal = current_probs_optimal_sorted[unique_indices]

    # calculate policy-ig = max entropy - current entropy
    policy_ig_values = max_H_for_N - final_target_entropies

    # plot thin line
    ax.plot(policy_ig_values, final_probs_optimal, linestyle='-', linewidth=1.5,
            color=colors[color_idx], label=f"{label}", alpha=0.7)

    line_data[label] = {
        'policy_ig': policy_ig_values,
        'probs': final_probs_optimal,
        'color': colors[color_idx]
    }
    color_idx = (color_idx + 1) % len(colors)

for label, data in line_data.items():
    segment_indices = np.where((data['probs'] >= 0.15) & (data['probs'] <= 0.95))[0]
    if len(segment_indices) > 0:
        segment_start_idx = segment_indices[0]
        segment_end_idx = segment_indices[-1]
        ax.plot(data['policy_ig'][segment_start_idx:segment_end_idx + 1],
                data['probs'][segment_start_idx:segment_end_idx + 1],
                linestyle='-', linewidth=3.0, color=data['color'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(direction='in', length=4, width=1)

ax.set_xlabel("Policy-IG (nats)", fontsize=14, family='Arial')
ax.set_ylabel("Probability of initiating 'best' action", fontsize=14, family='Arial')

ax.set_ylim(-0.05, 1.05)
all_max_entropies = [np.log(n) if n > 1 else 0 for n in action_counts.values()]
max_policy_ig = max(all_max_entropies) if all_max_entropies else 1.0
ax.set_xlim(-0.1, max_policy_ig * 1.05 if max_policy_ig > 0 else 1.0)

ax.legend(loc='center right', fontsize=10, frameon=False)
plt.tight_layout()
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('policy_ig_shift_action_initiation.pdf', bbox_inches='tight', dpi=300)
plt.show()