import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import random
import matplotlib as mpl

random.seed(0)
np.random.seed(0)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.constrained_layout.use': True
})

#sim params
n_timesteps = 150
t = np.arange(n_timesteps)
window_size = 12
cognitive_learning_rate = 1.0

# ANCCR-like model parameters
eligibility_t_constant_anccr = 10.0
eligibility_decay_anccr = np.exp(-1.0 / eligibility_t_constant_anccr)
alpha_M_anccr = 0.05
alpha_P_anccr = 0.1
alpha_CW_anccr = 0.1
rpe_stimulus_threshold = 1.0
num_rpe_stimuli_anccr = 2

random.seed(0)
np.random.seed(0)

# phase definitions
phases = {
    'Explore1': (0, 35),
    'ComplexFood': (35, 65),
    'Extended Rest': (65, 90),
    'Rapid Travel': (90, 115),
    'SimpleFood': (115, 135),
    'Return': (135, 150)
}
phase_names = list(phases.keys())

velocity = np.zeros(n_timesteps)
state_value = np.zeros(n_timesteps)
reward = np.zeros(n_timesteps)
rpe = np.zeros(n_timesteps)
n_actions = np.zeros(n_timesteps, dtype=int)
entropy = np.zeros(n_timesteps)
entropy_multiplier = np.zeros(n_timesteps)

E_traces_anccr = np.zeros(num_rpe_stimuli_anccr)
M_stimulus_anccr = np.zeros(num_rpe_stimuli_anccr)
M_reward_event_anccr = 0.0
P_stimulus_to_reward_anccr = np.zeros(num_rpe_stimuli_anccr)
S_stimulus_to_reward_anccr = np.zeros(num_rpe_stimuli_anccr)
CW_stimulus_to_reward_anccr = np.zeros(num_rpe_stimuli_anccr)
anccr_Q_trace = np.zeros(n_timesteps)
anccr_CW_update_trace = np.zeros(n_timesteps)

# define target velecoties
target_velocities = {'Explore1': 0.85, 'ComplexFood': 0.08, 'Extended Rest': 0.03, 'Rapid Travel': 0.9,
                     'SimpleFood': 0.45, 'Return': 0.15}
transition_duration = 8;
last_end_vel = 0;
last_vel_val = 0.1
for name, (start, end) in phases.items():
    target_vel = target_velocities[name]
    if start > 0:
        ramp_start = max(last_end_vel, start - transition_duration // 2);
        ramp_end = min(end, start + transition_duration // 2)
        if ramp_end > ramp_start: vel_at_ramp_start_init = velocity[
            ramp_start - 1] if ramp_start > 0 else last_vel_val; velocity[ramp_start:ramp_end] = np.linspace(
            vel_at_ramp_start_init, target_vel, ramp_end - ramp_start)
    main_start = start + transition_duration // 2 if start > 0 else start;
    main_end = end - transition_duration // 2 if name != 'Return' else end;
    main_start = min(main_start, n_timesteps);
    main_end = min(main_end, n_timesteps)
    if main_end > main_start: velocity[main_start:main_end] = np.random.normal(target_vel, 0.05, main_end - main_start)
    if start == 0 and main_start > 0: velocity[0:main_start] = np.linspace(last_vel_val, target_vel, main_start)
    if name != 'Return':
        next_phase_name_vel = list(phases.keys())[list(phases.keys()).index(name) + 1];
        next_target_vel = target_velocities[next_phase_name_vel];
        ramp_down_start = max(start, end - transition_duration // 2);
        ramp_down_end_initial = min(n_timesteps, end + transition_duration // 2);
        next_phase_start_vel, _ = phases[next_phase_name_vel];
        ramp_up_next_start = max(end, next_phase_start_vel - transition_duration // 2)
        if ramp_down_start < ramp_up_next_start: ramp_down_end = min(ramp_down_end_initial, ramp_up_next_start)
        if ramp_down_end > ramp_down_start: vel_at_ramp_start_trans = velocity[
            ramp_down_start - 1] if ramp_down_start > 0 else target_vel; velocity[
                                                                         ramp_down_start:ramp_down_end] = np.linspace(
            vel_at_ramp_start_trans, next_target_vel, ramp_down_end - ramp_down_start)
last_end_vel = end;
last_vel_val = velocity[end - 1] if end > 0 and end <= n_timesteps else target_vel
for name, (start, end) in phases.items():
    if start > 0 and end < n_timesteps and abs(velocity[start] - velocity[start - 1]) > 0.15:
        if start > 1 and start + 1 < n_timesteps: velocity[start - 1:start + 1] = np.linspace(velocity[start - 2],
                                                                                              velocity[start + 1], 3)
velocity = np.clip(velocity, 0, 1)

# State variable simulation
start, end = phases['Explore1'];
state_value[start:end] = np.linspace(2, 3.5, end - start) + np.random.normal(0, 0.25, end - start);
reward[start:end] = 0;
n_actions[start:end] = np.random.randint(25, 35, end - start);
rpe[start:end] = np.random.normal(0, 0.3, end - start)
explore_spike_time = 20;
explore_spike_reward = 5;
reward[explore_spike_time] = explore_spike_reward;
explore_spike_rpe_value = 5.0;
rpe[explore_spike_time] = explore_spike_rpe_value
if explore_spike_time + 1 < n_timesteps: state_value[explore_spike_time + 1:] += explore_spike_rpe_value * 0.4 * np.exp(
    -np.arange(n_timesteps - (explore_spike_time + 1)) * 0.1)
neg_surprise_time_explore = 10;
neg_surprise_rpe_explore = -3.5;
rpe[neg_surprise_time_explore] = neg_surprise_rpe_explore
if neg_surprise_time_explore + 1 < n_timesteps: state_value[
                                                neg_surprise_time_explore + 1:] += neg_surprise_rpe_explore * 0.3 * np.exp(
    -np.arange(n_timesteps - (neg_surprise_time_explore + 1)) * 0.15)
start, end = phases['ComplexFood'];
state_value_start_cf = state_value[start - 1] if start > 0 else 2.0;
complex_food_reward_time = 50;
complex_food_reward_value = 12;
state_value_target_at_reward = 10;
reward_time_rel = complex_food_reward_time - start;
duration_to_reward = reward_time_rel
if duration_to_reward > 0: state_value[start:complex_food_reward_time] = np.linspace(state_value_start_cf,
                                                                                     state_value_target_at_reward,
                                                                                     duration_to_reward) + np.random.normal(
    0, 0.3, duration_to_reward)
reward[complex_food_reward_time] = complex_food_reward_value;
n_actions[start:end] = np.random.randint(3, 5, end - start);
gamma = 0.95;
handling_rpe_noise_std = 0.8
for i in range(start, end):
    v_prev = state_value[i - 1] if i > 0 else state_value_start_cf
    if i < complex_food_reward_time and i >= start:
        v_curr = state_value[i]
        expected_v_increase = (state_value_target_at_reward - state_value_start_cf) / duration_to_reward if duration_to_reward > 0 else 0
        effort_cost = np.random.uniform(0.1, 0.4)
        base_rpe_val = reward[i] + gamma * v_curr - (v_prev + expected_v_increase) - effort_cost
        rpe[i] = base_rpe_val + np.random.normal(0, handling_rpe_noise_std)
    elif i == complex_food_reward_time:
        state_value[i] = state_value_target_at_reward + 2.0
        v_curr = state_value[i]
        rpe[i] = reward[i] + gamma * v_curr - v_prev
        rpe[i] = max(rpe[i], 6.0)
        state_value[i] = v_prev + rpe[i] * 0.3
        if i + 1 < end:
            decay_target = state_value[i] * 0.85
            num_steps_after_reward = end - (i + 1)
            if num_steps_after_reward > 0:
                state_value[i + 1:end] = np.linspace(state_value[i], decay_target, num_steps_after_reward) + np.random.normal(0, 0.2, num_steps_after_reward)
            elif i + 1 < n_timesteps:
                state_value[i + 1] = decay_target + np.random.normal(0, 0.2)
    elif i > complex_food_reward_time:
        if not (state_value[i] != 0 and i < end and i > complex_food_reward_time):
            v_curr_expected_decay = state_value[complex_food_reward_time] * (0.95 ** (i - complex_food_reward_time))
            state_value[i] = v_curr_expected_decay + np.random.normal(0, 0.2)
        v_curr = state_value[i]
        rpe[i] = reward[i] + gamma * v_curr - v_prev + np.random.normal(0, 0.3)
secondary_surprise_time_complex = 57
if complex_food_reward_time < secondary_surprise_time_complex < end: secondary_surprise_rpe_val = np.random.choice(
    [-3.0, 3.5]); reward[secondary_surprise_time_complex] = secondary_surprise_rpe_val * 0.3; v_prev_surprise = \
state_value[secondary_surprise_time_complex - 1]; state_value[
    secondary_surprise_time_complex] = v_prev_surprise + secondary_surprise_rpe_val * 0.5; rpe[
    secondary_surprise_time_complex] = secondary_surprise_rpe_val
if secondary_surprise_time_complex + 1 < n_timesteps: state_value[
                                                      secondary_surprise_time_complex + 1:] += secondary_surprise_rpe_val * 0.4 * np.exp(
    -np.arange(n_timesteps - (secondary_surprise_time_complex + 1)) * 0.12)
start, end = phases['Extended Rest'];
rest_start_value = state_value[start - 1] if start > 0 else state_value[
    phases['ComplexFood'][1] - 1 if phases['ComplexFood'][1] - 1 >= 0 else 0]
state_value[start:end] = np.linspace(rest_start_value, rest_start_value * 0.8, end - start) + np.random.normal(0, 0.1,
                                                                                                               end - start);
reward[start:end] = 0
n_actions[start:end] = np.random.randint(20, 30, end - start)
surprise_probability = 0.08
for i in range(start, end): v_prev = state_value[i - 1]; v_curr = state_value[i]; rpe[i] = reward[
                                                                                               i] + 0.95 * v_curr - v_prev
if np.random.rand() < surprise_probability: rpe[i] += np.random.normal(0, 1.5)
rest_surprise_time = 78
if start <= rest_surprise_time < end: rest_surprise_rpe_val = np.random.choice([-3.5, 4.0]); rpe[
    rest_surprise_time] = rest_surprise_rpe_val; state_value[rest_surprise_time] += rest_surprise_rpe_val * 0.3
if rest_surprise_time + 1 < n_timesteps: state_value[rest_surprise_time + 1:] += rest_surprise_rpe_val * 0.5 * np.exp(
    -np.arange(n_timesteps - (rest_surprise_time + 1)) * 0.1)
start, end = phases['Rapid Travel'];
state_value[start:end] = np.linspace(state_value[start - 1], 4, end - start) + np.random.normal(0, 0.5, end - start);
reward[start:end] = 0;
n_actions[start:start + transition_duration] = np.linspace(n_actions[start - 1] if start > 0 else 30, 8,
                                                           transition_duration).astype(int);
n_actions[start + transition_duration:end - transition_duration] = np.random.randint(5, 10, max(0, (
            end - transition_duration) - (start + transition_duration)));
n_actions[end - transition_duration:end] = np.linspace(8, n_actions[
    min(end, n_timesteps - 1)] if end < n_timesteps else 15, transition_duration).astype(int)
for i in range(start, end): v_prev = state_value[i - 1]; v_curr = state_value[i]; rpe[i] = reward[
                                                                                               i] + 0.95 * v_curr - v_prev + np.random.normal(
    0, 0.4)
travel_surprise_time = 102
if start <= travel_surprise_time < end: travel_surprise_rpe_val = -3.0; rpe[
    travel_surprise_time] = travel_surprise_rpe_val; state_value[travel_surprise_time] += travel_surprise_rpe_val * 0.3
if travel_surprise_time + 1 < n_timesteps: state_value[
                                           travel_surprise_time + 1:] += travel_surprise_rpe_val * 0.4 * np.exp(
    -np.arange(n_timesteps - (travel_surprise_time + 1)) * 0.12)
start, end = phases['SimpleFood'];
current_v_simple = state_value[start - 1] if start > 0 else 3.0;
n_actions[start:end] = np.random.randint(7, 12, end - start);
jackpot_time = 125
for i in range(start, end): is_reward_step = np.random.rand() < 0.35; v_prev = state_value[i - 1] if i > start else \
state_value[start - 1]
if i == jackpot_time:
    reward[i] = 7.0; current_v_simple += 4.5
elif is_reward_step:
    reward[i] = 1.5; current_v_simple += 1.0
else:
    reward[i] = 0; current_v_simple += 0.15
state_value[i] = current_v_simple + np.random.normal(0, 0.25);
rpe[i] = reward[i] + 0.95 * state_value[i] - v_prev
if i == jackpot_time: rpe[i] = max(rpe[i], 5.0)
start, end = phases['Return'];
state_value[start:end] = np.linspace(state_value[start - 1], state_value[start - 1] * 0.85,
                                     end - start) + np.random.normal(0, 0.15, end - start);
reward[start:end] = 0;
n_actions[start:end] = np.linspace(18, 8, end - start).astype(int)
for i in range(start, end): v_prev = state_value[i - 1]; v_curr = state_value[i]; rpe[i] = reward[
                                                                                               i] + 0.95 * v_curr - v_prev

target_entropy_multipliers = {'Explore1': 0.9, 'ComplexFood': 0.1, 'Extended Rest': 0.9, 'Rapid Travel': 0.1,
                              'SimpleFood': 0.45, 'Return': 0.55}
last_end_mult = 0
last_mult_val = 0.9
for name, (start_phase, end_phase) in phases.items():
    target_mult = target_entropy_multipliers[name]
    phase_len = end_phase - start_phase
    if phase_len <= 0:
        continue

    if start_phase > 0:
        ramp_start_mult = max(last_end_mult, start_phase - transition_duration // 2)
        ramp_end_mult = min(end_phase, start_phase + transition_duration // 2)

        if ramp_end_mult > ramp_start_mult:
            mult_at_ramp_start_init = entropy_multiplier[ramp_start_mult - 1] if ramp_start_mult > 0 else last_mult_val
            entropy_multiplier[ramp_start_mult:ramp_end_mult] = np.linspace(mult_at_ramp_start_init, target_mult,
                                                                            ramp_end_mult - ramp_start_mult)

    main_start_mult = start_phase + transition_duration // 2 if start_phase > 0 else start_phase
    main_end_mult = end_phase - transition_duration // 2 if name != 'Return' else end_phase
    main_start_mult = min(main_start_mult, n_timesteps)
    main_end_mult = min(main_end_mult, n_timesteps)

    if main_end_mult > main_start_mult:
        entropy_multiplier[main_start_mult:main_end_mult] = np.random.normal(target_mult, 0.05,
                                                                             main_end_mult - main_start_mult)

    if start_phase == 0 and main_start_mult > 0:
        entropy_multiplier[0:main_start_mult] = np.linspace(last_mult_val, target_mult, main_start_mult)

    if name != 'Return':
        next_phase_name_mult = list(phases.keys())[list(phases.keys()).index(name) + 1]
        next_target_mult = target_entropy_multipliers[next_phase_name_mult]
        ramp_down_start_mult_trans = max(start_phase, end_phase - transition_duration // 2)
        ramp_down_end_mult_initial = min(n_timesteps, end_phase + transition_duration // 2)
        next_phase_start_mult, _ = phases[next_phase_name_mult]
        ramp_up_next_start_mult = max(end_phase, next_phase_start_mult - transition_duration // 2)

        if ramp_down_start_mult_trans < ramp_up_next_start_mult:
            ramp_down_end_mult = min(ramp_down_end_mult_initial, ramp_up_next_start_mult)

            if ramp_down_end_mult > ramp_down_start_mult_trans:
                mult_at_ramp_start_trans = entropy_multiplier[
                    ramp_down_start_mult_trans - 1] if ramp_down_start_mult_trans > 0 else target_mult
                entropy_multiplier[ramp_down_start_mult_trans:ramp_down_end_mult] = np.linspace(
                    mult_at_ramp_start_trans, next_target_mult, ramp_down_end_mult - ramp_down_start_mult_trans)

    last_end_mult = end_phase
    last_mult_val = entropy_multiplier[end_phase - 1] if end_phase > 0 and end_phase <= n_timesteps else target_mult
entropy_multiplier = np.clip(entropy_multiplier, 0.01, 0.99)
for name, (start_phase, end_phase) in phases.items():
    if start_phase > 0 and end_phase < n_timesteps and abs(
            entropy_multiplier[start_phase] - entropy_multiplier[start_phase - 1]) > 0.15:
        if start_phase > 1 and start_phase + 1 < n_timesteps: entropy_multiplier[
                                                              start_phase - 1:start_phase + 1] = np.linspace(
            entropy_multiplier[start_phase - 2], entropy_multiplier[start_phase + 1], 3)
n_actions[n_actions < 1] = 1;
max_entropy = np.zeros_like(entropy);
valid_actions_mask = n_actions > 0;
max_entropy[valid_actions_mask] = np.log(n_actions[valid_actions_mask]);
entropy = max_entropy * entropy_multiplier;
entropy = np.minimum(entropy, max_entropy);
entropy = np.maximum(entropy, 0);
info_gain = max_entropy - entropy;
info_gain = np.maximum(info_gain, 0)

chunk_qualities = {name: 0.0 for name in phase_names};
active_chunk_quality_trace = np.zeros(n_timesteps);
current_phase_name_for_cq = phase_names[0]
for i_cq in range(n_timesteps):
    for name_cq, (start_cq, end_cq) in phases.items():
        if start_cq <= i_cq < end_cq: current_phase_name_for_cq = name_cq; break
    chunk_qualities[current_phase_name_for_cq] += cognitive_learning_rate * info_gain[i_cq]
    active_chunk_quality_trace[i_cq] = chunk_qualities[current_phase_name_for_cq]

# --- ANCCR-like causal learning sims
for i_time in range(n_timesteps):
    is_pos_rpe_event = rpe[i_time] > rpe_stimulus_threshold;
    is_neg_rpe_event = rpe[i_time] < -rpe_stimulus_threshold;
    active_rpe_stim_idx = -1
    if is_pos_rpe_event:
        active_rpe_stim_idx = 0
    elif is_neg_rpe_event:
        active_rpe_stim_idx = 1
    E_traces_anccr *= eligibility_decay_anccr
    if is_pos_rpe_event: E_traces_anccr[0] = 1.0
    if is_neg_rpe_event: E_traces_anccr[1] = 1.0
    M_stimulus_anccr[0] += alpha_M_anccr * (float(is_pos_rpe_event) - M_stimulus_anccr[0]);
    M_stimulus_anccr[1] += alpha_M_anccr * (float(is_neg_rpe_event) - M_stimulus_anccr[1]);
    M_stimulus_anccr = np.clip(M_stimulus_anccr, 1e-5, 1.0)
    is_actual_reward_event = reward[i_time] > 1e-3;
    M_reward_event_anccr += alpha_M_anccr * (float(is_actual_reward_event) - M_reward_event_anccr);
    M_reward_event_anccr = np.clip(M_reward_event_anccr, 1e-5, 1.0)
    current_da_proxy_anccr = rpe[i_time]
    if is_actual_reward_event:
        for s_idx in range(num_rpe_stimuli_anccr): P_stimulus_to_reward_anccr[s_idx] += alpha_P_anccr * (
                    E_traces_anccr[s_idx] - P_stimulus_to_reward_anccr[s_idx]); P_stimulus_to_reward_anccr[
            s_idx] = np.clip(P_stimulus_to_reward_anccr[s_idx], 0, 1)
    for s_idx in range(num_rpe_stimuli_anccr):
        if M_stimulus_anccr[s_idx] > 1e-5:
            S_stimulus_to_reward_anccr[s_idx] = P_stimulus_to_reward_anccr[s_idx] * (
                        M_reward_event_anccr / M_stimulus_anccr[s_idx])
        else:
            S_stimulus_to_reward_anccr[s_idx] = 0
        S_stimulus_to_reward_anccr[s_idx] = np.clip(S_stimulus_to_reward_anccr[s_idx], 0, 1)
    delta_cw_sum_for_active_stims = 0.0
    for s_idx in range(num_rpe_stimuli_anccr):
        target_cw_anccr = 1.0 if current_da_proxy_anccr >= 0 else 0.0;
        error_cw_anccr = target_cw_anccr - CW_stimulus_to_reward_anccr[s_idx];
        delta_this_cw_anccr = alpha_CW_anccr * error_cw_anccr * E_traces_anccr[s_idx]
        CW_stimulus_to_reward_anccr[s_idx] += delta_this_cw_anccr;
        CW_stimulus_to_reward_anccr[s_idx] = np.clip(CW_stimulus_to_reward_anccr[s_idx], 0, 1)
        if (s_idx == 0 and is_pos_rpe_event) or (
                s_idx == 1 and is_neg_rpe_event): delta_cw_sum_for_active_stims += delta_this_cw_anccr
    if active_rpe_stim_idx != -1:
        anccr_Q_trace[i_time] = S_stimulus_to_reward_anccr[active_rpe_stim_idx] * CW_stimulus_to_reward_anccr[
            active_rpe_stim_idx]
    else:
        anccr_Q_trace[i_time] = 0
    anccr_CW_update_trace[i_time] = delta_cw_sum_for_active_stims

df_for_corr = pd.DataFrame({
    'RPE': rpe, 'Value': state_value, 'Velocity': velocity, 'InfoGain': info_gain,
    'ANCCRCWUpdate': anccr_CW_update_trace
})

roll_corr_ig_rpe_orig = df_for_corr['InfoGain'].rolling(window=window_size, min_periods=1).corr(
    df_for_corr['RPE']).fillna(0)
roll_corr_ig_value_orig = df_for_corr['InfoGain'].rolling(window=window_size, min_periods=1).corr(
    df_for_corr['Value']).fillna(0)
roll_corr_ig_velocity_orig = df_for_corr['InfoGain'].rolling(window=window_size, min_periods=1).corr(
    df_for_corr['Velocity']).fillna(0)


roll_corr_ig_anccr_update = df_for_corr['InfoGain'].rolling(window=window_size, min_periods=1).corr(
    df_for_corr['ANCCRCWUpdate']).fillna(0)

# plotting
plt.rcParams['figure.constrained_layout.use'] = False
fig = plt.figure(figsize=(5, 8))
gs = fig.add_gridspec(31, 1)
axs = [
    fig.add_subplot(gs[0:3, 0]), fig.add_subplot(gs[3:7, 0]), fig.add_subplot(gs[7:11, 0]),
    fig.add_subplot(gs[11:15, 0]), fig.add_subplot(gs[16:21, 0]),
    fig.add_subplot(gs[22:26, 0]), fig.add_subplot(gs[27:31, 0])  # axs[5] and axs[6]
]
panel_colors = {
    'velocity': '#663399', 'state_value': '#0072B2', 'rpe_pos': 'forestgreen', 'rpe_neg': 'firebrick',
    'info_gain_line': '#000000', 'corr_rpe_dom': '#FF6347', 'corr_value_dom': '#4682B4',
    'corr_velocity_dom': '#3CB371', 'active_chunk_quality_line': 'darkorange', 'anccr_q_line': 'purple',
    'rpe_update_line': 'firebrick', 'infogain_update_line': '#000000', 'anccr_cw_update_line': 'teal',
    'corr_ig_rpe_update': '#FF6347',
    'corr_ig_anccr_update': '#9370DB'
}


def add_phase_lines_custom(ax, phases_dict, n_timesteps_plot, is_main_ig_ax=False):
    for i_phase_line, (name_p_line, (start_p_line, end_p_line)) in enumerate(phases_dict.items()):
        if i_phase_line > 0: ax.axvline(start_p_line, color='#666666', linestyle=':', alpha=0.7, linewidth=0.75)
        if is_main_ig_ax:
            mid_point_line = start_p_line + (end_p_line - start_p_line) / 2;
            label_name_line = name_p_line
            if label_name_line == 'ComplexFood':
                label_name_line = 'Complex food'
            elif label_name_line == 'Extended Rest':
                label_name_line = 'Rest'
            elif label_name_line == 'Rapid Travel':
                label_name_line = 'Travel'
            elif label_name_line == 'SimpleFood':
                label_name_line = 'Simple food'
            elif label_name_line == 'Explore1':
                label_name_line = 'Explore'
            ax.text(mid_point_line, ax.get_ylim()[1] * 1.05, label_name_line, ha='center', va='bottom', fontsize=7,
                    rotation=0, bbox=dict(facecolor='white', alpha=0.0, edgecolor='none', pad=0.1))
    ax.axvline(n_timesteps_plot, color='#666666', linestyle=':', alpha=0.7, linewidth=0.75)


for i_ax_fig, ax_fig_curr in enumerate(axs):
    ax_fig_curr.spines['top'].set_visible(False);
    ax_fig_curr.spines['right'].set_visible(False)
    ax_fig_curr.tick_params(axis='both', which='major', labelsize=8)
    for spine_fig in ax_fig_curr.spines.values(): spine_fig.set_linewidth(0.75)
    if i_ax_fig < len(axs) - 1: ax_fig_curr.set_xticklabels([])

ax = axs[0];
ax.plot(t, velocity, label='Velocity', color=panel_colors['velocity'], linewidth=1.5);
ax.set_ylabel('Velocity', fontsize=9);
ax.set_ylim(0, 1);
add_phase_lines_custom(ax, phases, n_timesteps);
ax.legend(loc='upper left', frameon=False, fontsize=7)
ax = axs[1];
ax.plot(t, state_value, label='State Value (Ext.)', color=panel_colors['state_value'], linewidth=1.5);
ax.set_ylabel('State Value', fontsize=9);
add_phase_lines_custom(ax, phases, n_timesteps);
ax.legend(loc='upper left', frameon=False, fontsize=7)
ax = axs[2];
rpe_bar_cols_p = [panel_colors['rpe_pos'] if r_val >= 0 else panel_colors['rpe_neg'] for r_val in rpe];
ax.bar(t, rpe, color=rpe_bar_cols_p, width=0.8, alpha=0.9);
ax.axhline(y=0, color='black', linewidth=0.5);
ax.set_ylabel('RPE', fontsize=9);
rpe_min_p, rpe_max_p = np.min(rpe), np.max(rpe);
ylim_pad_p = (rpe_max_p - rpe_min_p) * 0.1;
ax.set_ylim(rpe_min_p - ylim_pad_p, rpe_max_p + ylim_pad_p);
gp_p = mpatches.Patch(color=panel_colors['rpe_pos'], label='RPE ≥ 0', alpha=0.9);
rp_p = mpatches.Patch(color=panel_colors['rpe_neg'], label='RPE < 0', alpha=0.9);
ax.legend(handles=[gp_p, rp_p], loc='upper right', frameon=False, fontsize=7);
add_phase_lines_custom(ax, phases, n_timesteps)
ax = axs[3];
ax.fill_between(t, info_gain, 0, color='#E5E5E5', alpha=0.7);
ax.plot(t, info_gain, label='Information Gain', color=panel_colors['info_gain_line'], linewidth=1.5);
ax.set_ylabel('Info Gain (nats)', fontsize=9);
ccm_dom_p = {'RPE': panel_colors['corr_rpe_dom'], 'Value': panel_colors['corr_value_dom'],
             'Velocity': panel_colors['corr_velocity_dom']};
var_corr_p = ['RPE', 'Value', 'Velocity'];
cdf_dom_p = pd.DataFrame(
    {'RPE': roll_corr_ig_rpe_orig, 'Value': roll_corr_ig_value_orig, 'Velocity': roll_corr_ig_velocity_orig});
mct_p = 0.2;
macd_p = 1
ssi_p = window_size - 1;
sei_p = n_timesteps
for i_cp in range(ssi_p, sei_p):
    if i_cp >= len(cdf_dom_p): continue
    cc_p = cdf_dom_p.iloc[i_cp];
    mcv_p = 0;
    dvc_p = None
    for vc_p in var_corr_p:
        if pd.notna(cc_p[vc_p]) and abs(cc_p[vc_p]) > abs(mcv_p):
            if cc_p[vc_p] > mcv_p: mcv_p = cc_p[vc_p]; dvc_p = vc_p
    if dvc_p is not None and mcv_p > mct_p: cv_p = ccm_dom_p[dvc_p]; av_p = macd_p * (mcv_p - mct_p) / (
                1.0 - mct_p); av_p = np.clip(av_p, 0, macd_p); ax.axvspan(i_cp - 0.5, i_cp + 0.5, facecolor=cv_p,
                                                                          alpha=av_p, edgecolor=None, linewidth=0)
ledc_p = [Patch(facecolor=ccm_dom_p['RPE'], alpha=macd_p * 0.9, label='IG-RPE'),
          Patch(facecolor=ccm_dom_p['Value'], alpha=macd_p * 0.9, label='IG-Value'),
          Patch(facecolor=ccm_dom_p['Velocity'], alpha=macd_p * 0.9, label='IG-Velocity')];
llim_p = ax.legend(loc='upper left', frameon=False, fontsize=7);
ax.add_artist(llim_p);
ax.legend(handles=ledc_p, loc='upper right', title=f"Dom.Pos.Corr(r>{mct_p})", frameon=False, fontsize=6,
          title_fontsize=7);
add_phase_lines_custom(ax, phases, n_timesteps, is_main_ig_ax=True)

ax = axs[4];
ax.plot(t, roll_corr_ig_rpe_orig, label='Corr(IG,RPE)', color=panel_colors['corr_rpe_dom'], linewidth=1.5);
ax.plot(t, roll_corr_ig_value_orig, label='Corr(IG,Value)', color=panel_colors['corr_value_dom'], linewidth=1.5);
ax.plot(t, roll_corr_ig_velocity_orig, label='Corr(IG,Velocity)', color=panel_colors['corr_velocity_dom'],
        linewidth=1.5);
ax.set_ylabel(f'Rolling Corrs w/ IG\n(win={window_size})', fontsize=9);
ax.axhline(0, color='black', linewidth=0.5, linestyle='--');
ax.set_ylim(-1, 1);
add_phase_lines_custom(ax, phases, n_timesteps);
ax.legend(loc='lower left', frameon=False, ncol=1, fontsize=7)

ax = axs[5]
ax.plot(t, info_gain, label='InfoGain (Cog. Update)', color=panel_colors['info_gain_line'], linewidth=1.5)
ax.set_ylabel('InfoGain (Cog. Update)\n& Learn Sig. Corrs', fontsize=9)
add_phase_lines_custom(ax, phases, n_timesteps)

min_corr_thresh_ax5 = 0.2
max_alpha_ax5 = 0.4
shade_start_idx_ax5 = window_size - 1
shade_end_idx_ax5 = n_timesteps

correlations_ig = pd.DataFrame({
    'RPE_Update': roll_corr_ig_rpe_orig,  # Corr(IG, RPE)
    'ANCCR_Update': roll_corr_ig_anccr_update  # Corr(IG, ANCCR_CW_Update)
})
corr_vars_ax5 = ['RPE_Update', 'ANCCR_Update']
corr_colors_ax5 = {
    'RPE_Update': panel_colors['corr_ig_rpe_update'],
    'ANCCR_Update': panel_colors['corr_ig_anccr_update']
}

for i_corr_ax5 in range(shade_start_idx_ax5, shade_end_idx_ax5):
    if i_corr_ax5 >= len(correlations_ig): continue
    current_corrs_ax5 = correlations_ig.iloc[i_corr_ax5]
    max_corr_val_ax5 = 0
    dominant_var_ax5 = None
    for var_ax5 in corr_vars_ax5:
        if pd.notna(current_corrs_ax5[var_ax5]) and current_corrs_ax5[
            var_ax5] > max_corr_val_ax5:  # only positive dominant
            max_corr_val_ax5 = current_corrs_ax5[var_ax5]
            dominant_var_ax5 = var_ax5

    if dominant_var_ax5 is not None and max_corr_val_ax5 > min_corr_thresh_ax5:
        color_val_ax5 = corr_colors_ax5[dominant_var_ax5]
        alpha_val_ax5 = max_alpha_ax5 * (max_corr_val_ax5 - min_corr_thresh_ax5) / (1.0 - min_corr_thresh_ax5)
        alpha_val_ax5 = np.clip(alpha_val_ax5, 0, max_alpha_ax5)
        ax.axvspan(i_corr_ax5 - 0.5, i_corr_ax5 + 0.5, facecolor=color_val_ax5, alpha=alpha_val_ax5, edgecolor=None,
                   linewidth=0)

line_legend_ax5 = ax.legend(loc='upper left', frameon=False, fontsize=7)  # For the InfoGain line
ax.add_artist(line_legend_ax5)
handles_corr_ax5 = [
    Patch(facecolor=corr_colors_ax5['RPE_Update'], alpha=max_alpha_ax5 * 0.9, label='IG ~ RPE Upd.'),
    Patch(facecolor=corr_colors_ax5['ANCCR_Update'], alpha=max_alpha_ax5 * 0.9, label='IG ~ ANCCR Upd.')
]
ax.legend(handles=handles_corr_ax5, loc='upper right', title=f"Dom.Pos.Corr (r>{min_corr_thresh_ax5})", frameon=False,
          fontsize=6, title_fontsize=7)
min_y_ax5_val = min(0, np.min(info_gain) - 0.1)
max_y_ax5_val = max(np.max(info_gain) + 0.1, 0.1)
ax.set_ylim(bottom=min_y_ax5_val, top=max_y_ax5_val)

ax = axs[6]
ax.plot(t, rpe, label='RPE (MFRL Update)', color=panel_colors['rpe_update_line'], linewidth=1.5, alpha=0.8)
ax.plot(t, info_gain, label='Info Gain (Cog. Update)', color=panel_colors['infogain_update_line'], linewidth=1.5,
        linestyle='--', alpha=0.8)
ax.plot(t, anccr_CW_update_trace, label=r'$\Delta$CW (ANCCR Update)', color=panel_colors['anccr_cw_update_line'],
        linewidth=1.5, linestyle=':', alpha=0.8)
ax.set_xlabel('Time step', fontsize=9);
ax.set_ylabel('Learning Update Signals', fontsize=9);
ax.axhline(0, color='black', linewidth=0.5, linestyle='-');
ax.legend(loc='upper right', frameon=False, fontsize=7);
add_phase_lines_custom(ax, phases, n_timesteps)
min_signal_ax6 = min(np.min(rpe), np.min(info_gain), np.min(anccr_CW_update_trace), -0.1);
max_signal_ax6 = max(np.max(rpe), np.max(info_gain), np.max(anccr_CW_update_trace), 0.1);
padding_signal_ax6 = (max_signal_ax6 - min_signal_ax6) * 0.1;
ax.set_ylim(min_signal_ax6 - padding_signal_ax6, max_signal_ax6 + padding_signal_ax6)

plt.subplots_adjust(left=0.15, right=0.95, top=0.97, bottom=0.05, hspace=0.9)
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('naturalistic_foraging.pdf',format='pdf')
plt.show()
