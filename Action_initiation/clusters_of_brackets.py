import numpy as np
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import gaussian_kde

np.random.seed(0)

def calculate_da_and_policy_entropy_nats(q_values, beta):
    if q_values is None or len(q_values) == 0:
        return 0, 0, 0
    q_values = np.array(q_values)
    n_actions_val = len(q_values)

    if n_actions_val <= 1:
        return 0, 0, 0

    h_max = np.log(n_actions_val)

    if beta < 1e-6:
        pi = np.ones(n_actions_val) / n_actions_val
    else:
        q_stable = q_values - np.max(q_values)
        scaled_q = np.clip(beta * q_stable, -700, 700)
        exp_q = np.exp(scaled_q)
        sum_exp_q = np.sum(exp_q)
        if sum_exp_q == 0 or not np.isfinite(sum_exp_q) or sum_exp_q < 1e-9:
            max_q_indices = np.where(q_values >= np.max(q_values) - 1e-6)[0]
            pi = np.zeros(n_actions_val)
            if len(max_q_indices) > 0:
                pi[max_q_indices] = 1.0 / len(max_q_indices)
            else:
                pi = np.ones(n_actions_val) / n_actions_val
        else:
            pi = exp_q / sum_exp_q

    if not np.isclose(np.sum(pi), 1.0) and np.sum(pi) > 1e-9:
        pi = pi / np.sum(pi)
    elif np.sum(pi) < 1e-9 and n_actions_val > 0:
        pi = np.ones(n_actions_val) / n_actions_val

    h_policy = scipy.stats.entropy(pi)
    if not np.isfinite(h_policy) or h_policy < 0:
        if np.any(pi > (1.0 - 1e-6)):
            h_policy = 0.0
        else:
            h_policy = h_max

    da = h_max - h_policy
    da_val = max(0, da) if np.isfinite(da) else 0
    return da_val, h_policy, h_max

def generate_q_step_variable(level_name_for_logic, time_step, current_n_steps_total,
                             num_actions_for_level, q_gap, noise_mult):
    n_steps_pre_post = 2
    main_seq_time = time_step - n_steps_pre_post
    n_steps_main_actual = current_n_steps_total - 2 * n_steps_pre_post

    is_pre = (time_step < n_steps_pre_post)
    is_post = (time_step >= n_steps_pre_post + n_steps_main_actual)

    # phase flags
    start_phase = (main_seq_time == 0)
    end_phase = (main_seq_time == n_steps_main_actual - 1)
    sub1_start = (main_seq_time == 1)
    sub1_exec = (main_seq_time >= 2 and main_seq_time <= 10)
    sub2_start = (main_seq_time == 15)
    sub2_exec = (main_seq_time >= 16 and main_seq_time <= 25)

    is_neutral_main_phase = False

    current_low_pref_mean = 1.0
    current_high_pref_mean = current_low_pref_mean + q_gap
    current_high_pref_std = 1.0 * noise_mult
    current_low_pref_std = 0.5 * noise_mult
    current_neutral_mean = 1.5
    current_neutral_std = 0.1 * noise_mult

    q = np.zeros(num_actions_for_level)
    pref_action_idx = -1

    if is_pre or is_post:
        q = np.random.normal(current_neutral_mean, current_neutral_std, num_actions_for_level)
    else:
        if level_name_for_logic == 'High':
            pref_action_idx = 0 if start_phase else 1 if end_phase else -1

        elif level_name_for_logic == 'Mid':
            pref_action_idx = 0 if sub1_start else 2 if sub2_start else -1

        elif level_name_for_logic == 'Low':
            if sub1_exec:
                pref_action_idx = ((main_seq_time - 2) % 3) + 1
                pref_action_idx = pref_action_idx % num_actions_for_level
            elif sub2_exec:
                pref_action_idx = ((main_seq_time - 16) % 2) * 2
                pref_action_idx = pref_action_idx % num_actions_for_level

        if pref_action_idx != -1:
            actual_pref_idx = pref_action_idx % num_actions_for_level
            for i in range(num_actions_for_level):
                q[i] = np.random.normal(current_high_pref_mean, current_high_pref_std) if i == actual_pref_idx \
                    else np.random.normal(current_low_pref_mean, current_low_pref_std)
        else:
            q = np.random.normal(current_neutral_mean, current_neutral_std, num_actions_for_level)
            is_neutral_main_phase = True

    return q, is_neutral_main_phase

def classify_da_profile(da_series_main_task, max_da_possible_for_classification):
    if not da_series_main_task.any(): return "Inactive Throughout"
    threshold = 0.30 * max_da_possible_for_classification
    if threshold <= 0: threshold = 1e-6
    n_main = len(da_series_main_task)

    start_idx_end = int(0.20 * n_main)
    middle_idx_start = start_idx_end
    middle_idx_end = int(0.80 * n_main)
    end_idx_start = middle_idx_end

    if n_main == 0: return "Inactive Throughout"
    if start_idx_end == 0 and n_main > 0: start_idx_end = 1
    if middle_idx_start >= middle_idx_end: middle_idx_start = middle_idx_end - 1 if middle_idx_end > 0 else 0
    if end_idx_start >= n_main: end_idx_start = n_main - 1 if n_main > 0 else 0

    start_segment = da_series_main_task[0:start_idx_end]
    middle_segment = da_series_main_task[middle_idx_start:middle_idx_end]
    end_segment = da_series_main_task[end_idx_start:n_main]

    is_start_high = np.mean(start_segment) > threshold if start_segment.size > 0 else False
    is_middle_high = np.mean(middle_segment) > threshold if middle_segment.size > 0 else False
    is_end_high = np.mean(end_segment) > threshold if end_segment.size > 0 else False

    if is_start_high and is_middle_high and is_end_high:
        return "Active Throughout"
    elif is_start_high and not is_middle_high and is_end_high:
        return "Beginning-End"
    elif is_start_high and not is_middle_high and not is_end_high:
        return "Start Only"
    elif not is_start_high and not is_middle_high and is_end_high:
        return "End Only"
    elif not is_start_high and is_middle_high and not is_end_high:
        return "Middle Only"
    elif is_start_high and is_middle_high and not is_end_high:
        return "Start-Middle"
    elif not is_start_high and is_middle_high and is_end_high:
        return "Middle-End"
    else:
        return "Inactive Throughout"

# params
n_steps_pre_post = 2
n_steps_main = 30
n_steps = n_steps_main + 2 * n_steps_pre_post
analysis_stage = 'Late Learning'

noise_multipliers = np.linspace(0.1, 3.0, 10)
q_value_gaps = np.linspace(2, 15, 10)
n_actions_to_test = [10, 50, 200, 1000]

fixed_beta_values = {
    'Late Learning': {'High': 5.0, 'Mid': 8.0, 'Low': 12.0}
}

analysis_levels_to_simulate = ['High', 'Mid', 'Low']

profile_colors = {
    "Active Throughout": "#0000FF",
    "Beginning-End": "#FF0000",
    "End Only": "#00AA00",
    "Start Only": "#FF8C00",
    "Middle-End": "#9400D3",
    "Inactive Throughout": "#808080",
    "Start-Middle": "#8B4513",
    "Middle Only": "#008B8B",
}

# run sims
all_plot_data = {level: [] for level in analysis_levels_to_simulate}

for current_analysis_level_name in analysis_levels_to_simulate:
    beta_current = fixed_beta_values[analysis_stage][current_analysis_level_name]

    for current_n_actions in n_actions_to_test:
        h_max_current_na = np.log(current_n_actions) if current_n_actions > 1 else 0

        nm_samples = 7
        qg_samples = 7
        nm_indices = np.linspace(0, len(noise_multipliers) - 1, nm_samples, dtype=int)
        qg_indices = np.linspace(0, len(q_value_gaps) - 1, qg_samples, dtype=int)

        for nm_idx_val in nm_indices:
            noise_mult = noise_multipliers[nm_idx_val]
            for qg_idx_val in qg_indices:
                q_gap = q_value_gaps[qg_idx_val]
                da_trajectory = np.zeros(n_steps)
                neutral_phase_entropies_main_task = []

                for t in range(n_steps):
                    q_vals, is_neutral_main_phase = generate_q_step_variable(
                        current_analysis_level_name, t, n_steps,
                        current_n_actions, q_gap, noise_mult)
                    da_val, h_policy_val, step_h_max = calculate_da_and_policy_entropy_nats(q_vals, beta_current)
                    da_trajectory[t] = da_val

                    is_main_task_step = (t >= n_steps_pre_post) and (t < n_steps - n_steps_pre_post)
                    if is_neutral_main_phase and is_main_task_step:
                        neutral_phase_entropies_main_task.append(h_policy_val)

                da_main_task = da_trajectory[n_steps_pre_post: n_steps - n_steps_pre_post]
                classification = classify_da_profile(da_main_task, h_max_current_na)
                avg_neutral_h_policy = np.mean(
                    neutral_phase_entropies_main_task) if neutral_phase_entropies_main_task else 0
                if h_max_current_na > 0:
                    norm_H = avg_neutral_h_policy / h_max_current_na
                    normalized_policy_IG = 1.0 - norm_H
                else:
                    normalized_policy_IG = 0.0

                all_plot_data[current_analysis_level_name].append({
                    'log_n_actions': np.log10(current_n_actions),
                    'policy_IG': normalized_policy_IG,
                    'classification': classification,
                })

# plot
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.titlesize': 12
})

fig, axes = plt.subplots(1, len(analysis_levels_to_simulate), figsize=(6, 6), sharey=True)

if len(analysis_levels_to_simulate) == 1:
    axes = [axes]

# get all classes
master_class_list = set()
for level_name in analysis_levels_to_simulate:
    for p_data in all_plot_data[level_name]:
        master_class_list.add(p_data['classification'])
all_class_types_globally = sorted(list(master_class_list))

for i, current_level_logic_name in enumerate(analysis_levels_to_simulate):
    ax = axes[i]
    current_plot_data = all_plot_data[current_level_logic_name]

    if not current_plot_data:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f"{current_level_logic_name}", fontsize=11, fontweight='bold')
        continue

    min_log_na = min(p['log_n_actions'] for p in current_plot_data)
    max_log_na = max(p['log_n_actions'] for p in current_plot_data)

    x_grid = np.linspace(min_log_na - 0.1, max_log_na + 0.1, 50)
    y_grid = np.linspace(0, 1.0, 50)
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)

    for class_type in all_class_types_globally:
        class_points_x = np.array([
            p['log_n_actions']
            for p in current_plot_data
            if p['classification'] == class_type
        ])
        class_points_y = np.array([
            p['policy_IG']
            for p in current_plot_data
            if p['classification'] == class_type
        ])

        color = profile_colors.get(class_type, 'gray')

        if len(class_points_x) >= 4:
            try:
                kernel = gaussian_kde(
                    np.vstack([class_points_x, class_points_y]),
                    bw_method=0.35
                )
                Z = kernel(np.vstack([X_mesh.ravel(), Y_mesh.ravel()]))
                Z = Z.reshape(X_mesh.shape)
                Z = Z / np.max(Z) if np.max(Z) > 0 else Z

                levels = np.linspace(0.2, 0.8, 3)

                ax.contourf(
                    X_mesh, Y_mesh, Z, levels=levels,
                    colors=[color] * len(levels), alpha=0.5
                )
                ax.contour(
                    X_mesh, Y_mesh, Z, levels=[levels[0]],
                    colors=[color], linewidths=1.0
                )

            except np.linalg.LinAlgError:
                ax.scatter(
                    class_points_x, class_points_y,
                    c=color, alpha=0.3, s=8, label='_nolegend_'
                )
        else:
            ax.scatter(
                class_points_x, class_points_y,
                c=color, alpha=0.3, s=8, label='_nolegend_'
            )

    if i == 0:
        ax.set_ylabel("Policy-IG (normalized)", fontsize=10)

    ax.set_xlabel("Number of Actions", fontsize=10)
    ax.set_title(f"{current_level_logic_name}", fontsize=11, fontweight='bold')

    log_na_ticks = np.log10(n_actions_to_test)
    ax.set_xticks(log_na_ticks)
    ax.set_xticklabels([f"{int(10 ** t)}" for t in log_na_ticks], rotation=45)

    ax.set_ylim(0, 1.05)
    ax.grid(False)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

fig.suptitle("Decisional Autonomy Profile Classification", fontsize=12, y=0.98)

plt.tight_layout(rect=[0, 0.18, 1, 0.95])
plt.subplots_adjust(wspace=0.05)

actual_classes_present = []
for level_data in all_plot_data.values():
    for item in level_data:
        if item['classification'] not in actual_classes_present:
            actual_classes_present.append(item['classification'])

actual_classes_present.sort()

legend_elements = [mpatches.Patch(color=profile_colors.get(ct, 'gray'), alpha=0.8, label=ct)
                   for ct in actual_classes_present]

fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, fontsize=9, frameon=True, edgecolor='black', borderaxespad=1.0)

fig.set_figheight(2.7)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

plt.savefig('da_profile_classification.pdf', bbox_inches='tight')
plt.show()