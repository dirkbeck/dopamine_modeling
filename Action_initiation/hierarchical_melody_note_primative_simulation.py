import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib.patches as patches
import matplotlib as mpl

np.random.seed(0)

def calculate_da_nats(q_values, beta):
    ##calculate dopamine signal from q-values and beta
    if q_values is None or len(q_values) == 0:
        return 0

    q_values = np.array(q_values)
    n_actions = len(q_values)

    if n_actions <= 1:
        return 0

    # max entropy in nats
    h_max = np.log(n_actions)

    if beta < 1e-6:
        pi = np.ones(n_actions) / n_actions
    else:
        q_stable = q_values - np.max(q_values)
        scaled_q = np.clip(beta * q_stable, -700, 700)
        exp_q = np.exp(scaled_q)
        sum_exp_q = np.sum(exp_q)

        if sum_exp_q == 0 or not np.isfinite(sum_exp_q) or sum_exp_q < 1e-9:
            max_q_indices = np.where(q_values >= np.max(q_values) - 1e-6)[0]
            pi = np.zeros(n_actions)
            if len(max_q_indices) > 0:
                pi[max_q_indices] = 1.0 / len(max_q_indices)
            else:
                pi = np.ones(n_actions) / n_actions
        else:
            pi = exp_q / sum_exp_q

    pi /= np.sum(pi)

    # current entropy
    h_optimal = scipy.stats.entropy(pi)
    if not np.isfinite(h_optimal) or h_optimal < 0:
        h_optimal = 0

    da = h_max - h_optimal
    return max(0, da) if np.isfinite(da) else 0

# simulation setup
n_steps_pre_post = 2
n_steps_main = 30
n_steps = n_steps_main + 2 * n_steps_pre_post

levels = ['High', 'Mid', 'Low']
stages = ['Early Learning', 'Late Learning']
n_actions = {'High': 10, 'Mid': 88, 'Low': 10000}
beta_values = {
    'Early Learning': {'High': 5.0, 'Mid': .5, 'Low': .1},
    'Late Learning': {'High': 5.0, 'Mid': 8.0, 'Low': 12.0}
}
q_dist_params = {
    'high_pref': (8.0, 1.0),
    'low_pref': (1.0, 0.5),
    'neutral': (1.5, 0.1)
}
early_high_neutral_std = 0.6

def generate_q_step(level, time_step, n_steps_total, stage):
    main_seq_time = time_step - n_steps_pre_post
    n_steps_main_actual = n_steps_total - 2 * n_steps_pre_post

    is_pre = (time_step < n_steps_pre_post)
    is_post = (time_step >= n_steps_pre_post + n_steps_main_actual)

    start_phase = (main_seq_time == 0)
    end_phase = (main_seq_time == n_steps_main_actual - 1)
    sub1_start = (main_seq_time == 1)
    sub1_exec = (main_seq_time >= 2 and main_seq_time <= 10)
    transition = (main_seq_time >= 11 and main_seq_time <= 14)
    sub2_start = (main_seq_time == 15)
    sub2_exec = (main_seq_time >= 16 and main_seq_time <= 25)

    N = n_actions[level]
    q = np.zeros(N)
    high_mean, high_std = q_dist_params['high_pref']
    low_mean, low_std = q_dist_params['low_pref']
    neut_mean, base_neut_std = q_dist_params['neutral']

    # neutral for pre/post phases
    if is_pre or is_post:
        q = np.random.normal(neut_mean, base_neut_std, N)
    elif level == 'High':
        pref_action_idx = 0 if start_phase else 1 if end_phase else -1
        if pref_action_idx != -1:
            for i in range(N):
                q[i] = np.random.normal(high_mean, high_std) if i == pref_action_idx else np.random.normal(low_mean, low_std)
        else:
            current_neut_std = early_high_neutral_std if stage == 'Early Learning' else base_neut_std
            q = np.random.normal(neut_mean, current_neut_std, N)
    elif level == 'Mid':
        pref_action_idx = 0 if sub1_start else 2 if sub2_start else -1
        if pref_action_idx != -1:
            for i in range(N):
                q[i] = np.random.normal(high_mean, high_std) if i == pref_action_idx else np.random.normal(low_mean, low_std)
        else:
            q = np.random.normal(neut_mean, base_neut_std, N)
    elif level == 'Low':
        pref_action_idx = -1
        if sub1_exec:
            pref_action_idx = (main_seq_time % 3) + 1
        elif sub2_exec:
            pref_action_idx = (main_seq_time % 2) * 2
        if pref_action_idx != -1:
            for i in range(N):
                q[i] = np.random.normal(high_mean, high_std) if i == pref_action_idx else np.random.normal(low_mean, low_std)
        else:
            q = np.random.normal(neut_mean, base_neut_std, N)
    return q

# generate data and calculate dopamine
da_results = {stage: {level: np.zeros(n_steps) for level in levels} for stage in stages}
for stage in stages:
    current_q_values = {level: [generate_q_step(level, t, n_steps, stage) for t in range(n_steps)] for level in levels}
    for level in levels:
        beta = beta_values[stage][level]
        for t in range(n_steps):
            q_step = current_q_values[level][t]
            da_value = calculate_da_nats(q_step, beta)
            da_results[stage][level][t] = da_value

# plotting
fig1, axes1 = plt.subplots(len(levels), len(stages), figsize=(5, 4), sharex=True, sharey=True)
fig1.suptitle('Simulated Dopamine (Info Gain, nats) with Hierarchical Phase Shading')

time_steps = np.arange(n_steps)

# phase boundaries
pre_range = (0, n_steps_pre_post - 1)
post_range = (n_steps - n_steps_pre_post, n_steps - 1)
song1_range = (n_steps_pre_post + 0, n_steps_pre_post + 14)
song2_range = (n_steps_pre_post + 15, n_steps_pre_post + 29)
motif1_range = (n_steps_pre_post + 1, n_steps_pre_post + 10)
motif2_range = (n_steps_pre_post + 15, n_steps_pre_post + 25)
notes1_steps = range(n_steps_pre_post + 2, n_steps_pre_post + 11)
notes2_steps = range(n_steps_pre_post + 16, n_steps_pre_post + 26)

bg_color = 'darkgrey'
song_color = 'lightblue'
motif_color = 'lightcoral'
note_color = 'lightgreen'

for i, level in enumerate(levels):
    for j, stage in enumerate(stages):
        ax = axes1[i, j]

        ax.axvspan(-0.5, n_steps - 0.5, color=bg_color, alpha=0.2, zorder=0)

        if level == 'High':
            ax.axvspan(song1_range[0] - 0.5, song1_range[1] + 0.5, color=song_color, alpha=0.4, zorder=1)
            ax.axvspan(song2_range[0] - 0.5, song2_range[1] + 0.5, color=song_color, alpha=0.6, zorder=1)
        elif level == 'Mid':
            ax.axvspan(motif1_range[0] - 0.5, motif1_range[1] + 0.5, color=motif_color, alpha=0.5, zorder=1)
            ax.axvspan(motif2_range[0] - 0.5, motif2_range[1] + 0.5, color=motif_color, alpha=0.7, zorder=1)
        elif level == 'Low':
            note_width = 0.5
            gap = (1.0 - note_width) / 1.5
            for t_note in notes1_steps:
                ax.axvspan(t_note - 0.5 + gap, t_note + 0.5 - gap, color=note_color, alpha=0.6, zorder=1)
            for t_note in notes2_steps:
                ax.axvspan(t_note - 0.5 + gap, t_note + 0.5 - gap, color=note_color, alpha=0.8, zorder=1)

        ax.fill_between(time_steps, da_results[stage][level], color='black', alpha=0.7, zorder=2)
        ax.plot(time_steps, da_results[stage][level], linestyle='-', linewidth=0.8, color='white', zorder=3)

        ax.set_title(f'{level} Level - {stage}')
        if i == len(levels) - 1:
            ax.set_xlabel('Time Step / Decision Point')
        if j == 0:
            ax.set_ylabel('Dopamine (Info Gain, nats)')

        max_da_possible_nats = np.log(n_actions[level]) if n_actions[level] > 1 else 1
        ax.set_ylim(0, max_da_possible_nats * 1.1)
        ax.set_xlim(-0.5, n_steps - 0.5)
        ax.grid(False)

legend_patches = [
    patches.Patch(color='black', alpha=0.7, label='Dopamine Signal (Info Gain)'),
    patches.Patch(color=song_color, alpha=0.5, label='Routine duration (e.g. play song)'),
    patches.Patch(color=motif_color, alpha=0.6, label='Sub-routine duration (e.g. play note)'),
    patches.Patch(color=note_color, alpha=0.7, label='Motor primative (e.g. move finger)')
]
fig1.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.03), fontsize='small')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.savefig("melody_note_primative.pdf", format='pdf')
plt.show()