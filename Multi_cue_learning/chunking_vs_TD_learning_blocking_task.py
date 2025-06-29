import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

class Policy:
    def __init__(self, p_lick_target_init=0.5):
        self.p_lick_target = p_lick_target_init
        self.p_lick_current = 0.5
        self._update_probabilities()

    def _update_probabilities(self):
        self.p_lick = np.clip(self.p_lick_current, 1e-9, 1 - 1e-9)
        self.p_no_lick = 1.0 - self.p_lick
        self.actions = ["Lick", "No Lick"]
        self.probabilities = np.array([self.p_lick, self.p_no_lick])
        self.entropy = self._calculate_entropy()

    def _calculate_entropy(self):
        if np.allclose(self.p_lick, 0) or np.allclose(self.p_lick, 1):
            return 0.0
        if self.p_lick <= 1e-9 or self.p_no_lick <= 1e-9:
            return 0.0
        return -(self.p_lick * np.log(self.p_lick + 1e-9) + self.p_no_lick * np.log(self.p_no_lick + 1e-9))

    def update_learning(self, learning_rate):
        self.p_lick_current += learning_rate * (self.p_lick_target - self.p_lick_current)
        self._update_probabilities()

    def get_current_p_lick_for_display(self):
        return self.p_lick_current


class Chunk:
    def __init__(self, name, cues, p_lick_target_for_policy=0.5):
        self.name = name
        self.cues = cues
        self.policy = Policy(p_lick_target_init=p_lick_target_for_policy)
        self.h_max = np.log(len(self.policy.actions)) if len(self.policy.actions) > 0 else 0
        self.q_intrinsic = 0
        self.is_learning_active = False
        self.update_q_intrinsic()

    def update_q_intrinsic(self):
        self.q_intrinsic = self.h_max - self.policy.entropy

    def learn_policy_step(self, learning_rate):
        if self.is_learning_active:
            self.policy.update_learning(learning_rate)
            self.update_q_intrinsic()


# params
H_MAX_GLOBAL = np.log(2)
EPISODES_PER_EPOCH = 40
TOTAL_EPOCHS = 4
POLICY_LEARNING_RATE_CHUNKER = 0.15
ALPHA_Q = 0.15
GAMMA_Q = 0.9
TEMPERATURE_Q = 1.0    # softmax temperature

STATE_CHEESE_NO_CUE = 0
STATE_RED_BELL_ONLY = 1
STATE_BLUE_BELL_ONLY = 2
STATE_RED_BLUE_COMPOUND = 3
NUM_STATES_Q = 4
ACTION_LICK_Q = 0
ACTION_NO_LICK_Q = 1
NUM_ACTIONS_Q = 2


def choose_action(q_values, temperature):
    # subtract max for numerical stability
    z = q_values / temperature
    z = z - np.max(z)
    exp_z = np.exp(z)
    probs = exp_z / np.sum(exp_z)
    return np.random.choice(NUM_ACTIONS_Q, p=probs)


def simulate_delayed_start_plot():
    chunk_dictionary = {}
    total_sim_episodes = TOTAL_EPOCHS * EPISODES_PER_EPOCH

    policy_baseline_chunk = Policy(p_lick_target_init=0.5)

    chunker_traces = {
        "CheeseOnly": [np.nan] * total_sim_episodes,
        "Red+cheese": [np.nan] * total_sim_episodes,
        "Red+blue+cheese": [np.nan] * total_sim_episodes
    }

    q_table = np.zeros((NUM_STATES_Q, NUM_ACTIONS_Q))
    qlearner_traces = {
        "V(S_Cheese)": [np.nan] * total_sim_episodes,
        "V(S_Red)": [np.nan] * total_sim_episodes,
        "V(S_BlueOnly)": [np.nan] * total_sim_episodes,
        "V(S_RedBlueCompound)": [np.nan] * total_sim_episodes
    }

    P_LICK_TARGET_UNIVERSAL = 0.95
    P_LICK_TARGET_CHEESE_ONLY = P_LICK_TARGET_UNIVERSAL
    P_LICK_TARGET_RED = P_LICK_TARGET_UNIVERSAL
    P_LICK_TARGET_RED_BLUE = P_LICK_TARGET_UNIVERSAL

    stim_epoch = 4
    stim_duration_episodes = int(EPISODES_PER_EPOCH * 0.5)
    stim_start_episode_global_idx = (stim_epoch - 1) * EPISODES_PER_EPOCH
    stim_end_episode_global_idx = stim_start_episode_global_idx + stim_duration_episodes

    global_episode_idx = 0
    for epoch in range(1, TOTAL_EPOCHS + 1):
        # chunker learning activation
        if epoch > 1 and "CheeseOnly" in chunk_dictionary:
            chunk_dictionary["CheeseOnly"].is_learning_active = False
        if epoch > 2 and "Red+cheese" in chunk_dictionary:
            chunk_dictionary["Red+cheese"].is_learning_active = False
        if "Red+blue+cheese" in chunk_dictionary:
            chunk_dictionary["Red+blue+cheese"].is_learning_active = False

        if epoch == 1:
            if "CheeseOnly" not in chunk_dictionary:
                chunk_dictionary["CheeseOnly"] = Chunk("CheeseOnly", {"Cheese"}, P_LICK_TARGET_CHEESE_ONLY)
            chunk_dictionary["CheeseOnly"].is_learning_active = True
        elif epoch == 2:
            if "Red+cheese" not in chunk_dictionary:
                chunk_dictionary["Red+cheese"] = Chunk("Red+cheese", {"RedBell"}, P_LICK_TARGET_RED)
            chunk_dictionary["Red+cheese"].is_learning_active = True
        elif epoch == 3:
            if "Red+blue+cheese" not in chunk_dictionary:
                chunk_dictionary["Red+blue+cheese"] = Chunk("Red+blue+cheese", {"RedBell", "BlueBell"},
                                                            P_LICK_TARGET_RED_BLUE)

        for episode_in_epoch in range(EPISODES_PER_EPOCH):
            # chunker logic
            if epoch == stim_epoch and stim_start_episode_global_idx <= global_episode_idx < stim_end_episode_global_idx and "Red+blue+cheese" in chunk_dictionary:
                chunk_dictionary["Red+blue+cheese"].is_learning_active = True
            elif "Red+blue+cheese" in chunk_dictionary and global_episode_idx >= stim_end_episode_global_idx:
                chunk_dictionary["Red+blue+cheese"].is_learning_active = False

            for chunk_obj in chunk_dictionary.values():
                chunk_obj.learn_policy_step(POLICY_LEARNING_RATE_CHUNKER)

            # record chunker traces
            if epoch >= 1 and "CheeseOnly" in chunk_dictionary:
                chunker_traces["CheeseOnly"][global_episode_idx] = chunk_dictionary["CheeseOnly"].q_intrinsic
            if epoch >= 2 and "Red+cheese" in chunk_dictionary:
                chunker_traces["Red+cheese"][global_episode_idx] = chunk_dictionary["Red+cheese"].q_intrinsic
            if epoch >= 3 and "Red+blue+cheese" in chunk_dictionary:
                chunker_traces["Red+blue+cheese"][global_episode_idx] = chunk_dictionary["Red+blue+cheese"].q_intrinsic

            # q-learner logic
            q_learner_current_trial_states = []
            enable_blue_to_red_propagation = False

            if epoch == 1:
                q_learner_current_trial_states.append(STATE_CHEESE_NO_CUE)
            elif epoch == 2:
                q_learner_current_trial_states.append(STATE_RED_BELL_ONLY)
            elif epoch == 3:
                rand_e3 = np.random.rand()
                if rand_e3 < 0.5:
                    q_learner_current_trial_states.append(STATE_BLUE_BELL_ONLY)
                else:
                    q_learner_current_trial_states.append(STATE_RED_BLUE_COMPOUND)
            elif epoch == 4:
                q_learner_current_trial_states.extend(
                    [STATE_BLUE_BELL_ONLY, STATE_RED_BLUE_COMPOUND, STATE_RED_BELL_ONLY])
                if global_episode_idx >= stim_start_episode_global_idx:
                    enable_blue_to_red_propagation = True

            if q_learner_current_trial_states:
                current_q_s = np.random.choice(q_learner_current_trial_states)
                action_q = choose_action(q_table[current_q_s], TEMPERATURE_Q)
                reward_q = 0
                next_s_max_q = 0

                if current_q_s == STATE_BLUE_BELL_ONLY and enable_blue_to_red_propagation:
                    s_blue = STATE_BLUE_BELL_ONLY
                    a_blue = action_q
                    r_blue = 0
                    s_next_after_blue = STATE_RED_BELL_ONLY
                    val_of_s_next_after_blue = np.max(q_table[s_next_after_blue])
                    q_table[s_blue, a_blue] += ALPHA_Q * (
                                r_blue + GAMMA_Q * val_of_s_next_after_blue - q_table[s_blue, a_blue])
                elif current_q_s == STATE_BLUE_BELL_ONLY and not enable_blue_to_red_propagation:
                    r_blue_only = 0
                    q_table[current_q_s, action_q] += ALPHA_Q * (
                                r_blue_only + GAMMA_Q * 0 - q_table[current_q_s, action_q])
                else:
                    if current_q_s == STATE_CHEESE_NO_CUE and action_q == ACTION_LICK_Q:
                        reward_q = 1.0
                    elif current_q_s == STATE_RED_BELL_ONLY and action_q == ACTION_LICK_Q:
                        reward_q = 1.0
                    elif current_q_s == STATE_RED_BLUE_COMPOUND and action_q == ACTION_LICK_Q:
                        reward_q = 1.0
                    q_table[current_q_s, action_q] += ALPHA_Q * (
                                reward_q + GAMMA_Q * next_s_max_q - q_table[current_q_s, action_q])

            # record Q traces
            if epoch >= 1:
                qlearner_traces["V(S_Cheese)"][global_episode_idx] = np.max(q_table[STATE_CHEESE_NO_CUE])
            if epoch >= 2:
                qlearner_traces["V(S_Red)"][global_episode_idx] = np.max(q_table[STATE_RED_BELL_ONLY])
            if epoch >= 3:
                qlearner_traces["V(S_BlueOnly)"][global_episode_idx] = np.max(q_table[STATE_BLUE_BELL_ONLY])
                qlearner_traces["V(S_RedBlueCompound)"][global_episode_idx] = np.max(q_table[STATE_RED_BLUE_COMPOUND])

            global_episode_idx += 1

    return chunker_traces, qlearner_traces, q_table


chunker_res_delay, qlearner_res_delay, final_q_delay = simulate_delayed_start_plot()

# plotting
total_eps_delay = TOTAL_EPOCHS * EPISODES_PER_EPOCH
eps_numbers_delay = np.arange(1, total_eps_delay + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))

colors = {'cheese': 'green', 'red': 'red', 'blue': 'blue', 'purple': 'purple'}

ax1.plot(eps_numbers_delay, chunker_res_delay["CheeseOnly"], color=colors['cheese'], linestyle='-', linewidth=2,
         label="Cheese chunk")
ax1.plot(eps_numbers_delay, chunker_res_delay["Red+cheese"], color=colors['red'], linestyle='-', linewidth=2,
         label="Red+cheese chunk")
ax1.plot(eps_numbers_delay, chunker_res_delay["Red+blue+cheese"], color=colors['purple'], linestyle='-', linewidth=2,
         label="Red+blue+cheese chunk")

ax1.plot(eps_numbers_delay, qlearner_res_delay["V(S_Cheese)"], color=colors['cheese'], linestyle='--', linewidth=2,
         alpha=0.7, label="Cheese value (TD algorithm)")
ax1.plot(eps_numbers_delay, qlearner_res_delay["V(S_Red)"], color=colors['red'], linestyle='--', linewidth=2, alpha=0.7,
         label="Red cue value (TD algorithm)")
ax1.plot(eps_numbers_delay, qlearner_res_delay["V(S_BlueOnly)"], color=colors['blue'], linestyle='--', linewidth=2,
         alpha=0.7, label="Blue cue value (TD algorithm)")

ax1.set_xlabel("Episode Number", fontsize=10)
ax1.set_ylabel('Value (chunk or cue)', color='black', fontsize=10)
ax1.tick_params(axis='y', labelsize=9)
ax1.tick_params(axis='x', labelsize=9)
ax1.set_ylim(-0.05, 1.10)

epoch_labels_delay = ["E1: Learn Cheese", "E2: Learn Red", "E3: Blue Intro", "E4: Stim. + Consolidate"]
stim_epoch_plot_delay = 4
stim_duration_plot_delay = int(EPISODES_PER_EPOCH * 0.5)
stim_band_plot_start_delay = (stim_epoch_plot_delay - 1) * EPISODES_PER_EPOCH + 0.5
stim_band_plot_end_delay = stim_band_plot_start_delay + stim_duration_plot_delay

for i in range(1, TOTAL_EPOCHS):
    xc = i * EPISODES_PER_EPOCH
    ax1.axvline(x=xc + 0.5, color='dimgray', linestyle=':', linewidth=1.0)

ax1.axvspan(stim_band_plot_start_delay, stim_band_plot_end_delay, color='yellow', alpha=0.3)

tick_pos_delay = [i * EPISODES_PER_EPOCH + EPISODES_PER_EPOCH / 2 for i in range(TOTAL_EPOCHS)]
valid_tick_pos_delay = [tp for tp in tick_pos_delay if tp <= total_eps_delay]
valid_tick_labels_delay = [epoch_labels_delay[i] for i, tp in enumerate(tick_pos_delay) if tp <= total_eps_delay]
ax1.set_xticks(valid_tick_pos_delay)
ax1.set_xticklabels(valid_tick_labels_delay, rotation=10, ha="right", fontsize=8)

import matplotlib.patches as mpatches

handles, labels = ax1.get_legend_handles_labels()
stim_patch = mpatches.Patch(color='yellow', alpha=0.3, label='Artificial stimulation')
if 'Artificial stimulation' not in labels:
    handles.append(stim_patch)
    labels.append('Artificial stimulation')

desired_order_map = {
    "Cheese chunk": 0, "Red+cheese chunk": 1, "Red+blue+cheese chunk": 2,
    "Cheese value (TD algorithm)": 3, "Red cue value (TD algorithm)": 4,
    "Blue cue value (TD algorithm)": 5, "Artificial stimulation": 6
}
sorted_handles_labels = sorted(zip(handles, labels), key=lambda x: desired_order_map.get(x[1], 99))
if sorted_handles_labels:
    sorted_handles, sorted_labels = zip(*sorted_handles_labels)
    ax1.legend(sorted_handles, sorted_labels, fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))
else:
    ax1.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.02, 0.5))

ax1.grid(False)
fig.tight_layout(rect=[0, 0, 0.75, 1])
plt.rcParams['pdf.fonttype'] = 42
plt.savefig('chunking_vs_TD_learning_blocking_task.pdf')
plt.show()