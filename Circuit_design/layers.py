import numpy as np
from collections import deque


class EnvironmentalStimuliLayer:
    def __init__(self, grid_size=5, memory_size=5):
        self.grid_size = grid_size
        self.memory_size = memory_size
        self.state_memory = deque(maxlen=memory_size)
        self.feature_size = 8
        self.activation_count = 0

    def reset(self):
        self.state_memory.clear()

    def extract_features(self, state):
        row, col = state
        norm_row = row / (self.grid_size - 1)
        norm_col = col / (self.grid_size - 1)

        dist_to_top = row
        dist_to_bottom = self.grid_size - 1 - row
        dist_to_left = col
        dist_to_right = self.grid_size - 1 - col

        norm_dist_top = dist_to_top / (self.grid_size - 1)
        norm_dist_bottom = dist_to_bottom / (self.grid_size - 1)
        norm_dist_left = dist_to_left / (self.grid_size - 1)
        norm_dist_right = dist_to_right / (self.grid_size - 1)

        region_boundary = self.grid_size // 2
        in_exploit_region = 1.0 if col < region_boundary else 0.0

        if col < region_boundary:
            dist_to_region_boundary = region_boundary - col
        else:
            dist_to_region_boundary = col - region_boundary + 1
        norm_dist_region = dist_to_region_boundary / self.grid_size

        features = np.array([
            norm_row, norm_col,
            norm_dist_top, norm_dist_bottom, norm_dist_left, norm_dist_right,
            in_exploit_region,
            norm_dist_region
        ])

        return features

    def process(self, state):
        features = self.extract_features(state)
        self.state_memory.append(state)

        motion_features = np.zeros(2)
        if len(self.state_memory) >= 2:
            prev_state = self.state_memory[-2]
            curr_state = self.state_memory[-1]
            motion_features[0] = curr_state[0] - prev_state[0]
            motion_features[1] = curr_state[1] - prev_state[1]

        combined_features = np.concatenate([features, motion_features])
        self.activation_count += 1
        return combined_features


class ShortTermPlanningLayer:
    def __init__(self, num_actions=4, memory_size=10):
        self.num_actions = num_actions
        self.memory_size = memory_size
        self.state_memory = deque(maxlen=memory_size)
        self.action_memory = deque(maxlen=memory_size)
        self.reward_memory = deque(maxlen=memory_size)
        self.short_term_q = np.zeros((25, num_actions))
        self.alpha = 0.3
        self.gamma = 0.7
        self.activation_count = 0
        self.td_errors = []

    def reset(self):
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()

    def remember(self, state_idx, action, reward):
        self.state_memory.append(state_idx)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

        if len(self.state_memory) >= 2:
            s = self.state_memory[-2]
            a = self.action_memory[-2]
            r = self.reward_memory[-1]
            s_next = self.state_memory[-1]

            best_next = np.max(self.short_term_q[s_next])
            td_error = r + self.gamma * best_next - self.short_term_q[s, a]
            self.short_term_q[s, a] += self.alpha * td_error
            self.td_errors.append(abs(td_error))

    def process(self, features, state_idx):
        row_pos, col_pos = features[0:2]

        up_bias = 0.5 - row_pos
        down_bias = row_pos - 0.5
        left_bias = 0.5 - col_pos
        right_bias = col_pos - 0.5

        action_preferences = np.array([up_bias, down_bias, left_bias, right_bias])
        action_preferences = 0.3 * action_preferences + 0.7 * self.short_term_q[state_idx]

        self.activation_count += 1
        return action_preferences


class LongTermPlanningLayer:
    def __init__(self, num_actions=4, planning_horizon=3):
        self.num_actions = num_actions
        self.planning_horizon = planning_horizon
        self.transition_counts = np.zeros((25, num_actions, 25))
        self.reward_sums = np.zeros((25, num_actions))
        self.action_counts = np.zeros((25, num_actions))
        self.value_estimates = np.zeros(25)
        self.gamma = 0.9
        self.activation_count = 0
        self.model_updates = 0
        self.value_changes = []
        self.uncertainty_estimates = np.ones(25) * 0.5

    def update_model(self, state, action, next_state, reward):
        self.transition_counts[state, action, next_state] += 1
        self.reward_sums[state, action] += reward
        self.action_counts[state, action] += 1
        self.model_updates += 1

        count = self.action_counts[state, action]
        self.uncertainty_estimates[state] = max(0.1, 1.0 / (1.0 + 0.1 * count))

    def get_transition_probs(self, state, action):
        counts = self.transition_counts[state, action]
        total = np.sum(counts)
        if total > 0:
            return counts / total
        else:
            return np.ones(25) / 25

    def get_expected_reward(self, state, action):
        count = self.action_counts[state, action]
        if count > 0:
            return self.reward_sums[state, action] / count
        else:
            return 0.0

    def value_iteration_step(self):
        old_values = self.value_estimates.copy()
        new_values = np.zeros(25)
        for s in range(25):
            action_values = []
            for a in range(self.num_actions):
                expected_reward = self.get_expected_reward(s, a)
                next_state_probs = self.get_transition_probs(s, a)
                expected_next_value = np.sum(next_state_probs * self.value_estimates)
                action_values.append(expected_reward + self.gamma * expected_next_value)

            new_values[s] = max(action_values) if action_values else 0.0

        value_change = np.mean(np.abs(new_values - old_values))
        self.value_changes.append(value_change)
        self.value_estimates = new_values

    def plan(self, state_idx):
        action_values = np.zeros(self.num_actions)

        for _ in range(3):
            self.value_iteration_step()

        for a in range(self.num_actions):
            immediate_reward = self.get_expected_reward(state_idx, a)
            next_state_probs = self.get_transition_probs(state_idx, a)
            next_state_value = np.sum(next_state_probs * self.value_estimates)
            action_values[a] = immediate_reward + self.gamma * next_state_value

        self.activation_count += 1
        return action_values, self.uncertainty_estimates[state_idx]


class ConflictResolutionLayer:
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        self.st_confidence = 0.5
        self.lt_confidence = 0.5
        self.uncertainty_memory = deque(maxlen=100)
        self.activation_count = 0
        self.conflict_levels = []

    def update_confidence(self, short_term_error, long_term_error):
        total_error = short_term_error + long_term_error
        if total_error > 0:
            self.st_confidence = long_term_error / total_error
            self.lt_confidence = short_term_error / total_error

        uncertainty = min(short_term_error, long_term_error)
        self.uncertainty_memory.append(uncertainty)

    def get_conflict_level(self, stp_values, ltp_values):
        stp_best = np.argmax(stp_values)
        ltp_best = np.argmax(ltp_values)

        if stp_best == ltp_best:
            return 0.0
        else:
            stp_preference = stp_values[stp_best] - stp_values[ltp_best]
            ltp_preference = ltp_values[ltp_best] - ltp_values[stp_best]

            conflict_strength = (stp_preference + ltp_preference) / 2.0
            return min(1.0, conflict_strength)

    def process(self, stp_values, ltp_values, ltp_uncertainty):
        conflict_level = self.get_conflict_level(stp_values, ltp_values)
        self.conflict_levels.append(conflict_level)

        current_uncertainty = np.mean(self.uncertainty_memory) if self.uncertainty_memory else 0.5

        combined_uncertainty = (0.4 * conflict_level +
                                0.4 * ltp_uncertainty +
                                0.2 * current_uncertainty)

        combined_values = (self.st_confidence * stp_values + self.lt_confidence * ltp_values)

        if conflict_level > 0.3 and combined_uncertainty > 0.4:
            random_component = 0.3
            random_values = np.random.random(self.num_actions)
            combined_values = (1 - random_component) * combined_values + random_component * random_values

        self.activation_count += 1
        return combined_values, conflict_level, combined_uncertainty


class MotorProgramsLayer:
    def __init__(self, num_actions=4, num_primitives=2):
        self.num_actions = num_actions
        self.num_primitives = num_primitives
        self.primitives = [
            [0, 3, 3],  # Go up then right twice
            [1, 2, 2],  # Go down then left twice
        ]
        self.current_primitive = None
        self.primitive_step = 0
        self.activation_count = 0
        self.primitive_usage_count = 0
        self.action_selection = np.zeros(num_actions)

    def translate_to_state_index(self, features):
        row_pos, col_pos = features[0:2]

        row_idx = min(4, max(0, int(row_pos * 5)))
        col_idx = min(4, max(0, int(col_pos * 5)))

        base_index = row_idx * 5 + col_idx
        return base_index

    def decide_primitive_use(self, action_values, conflict_level, exploration_signal):
        action_clarity = np.max(action_values) - np.mean(action_values)

        primitive_probability = 0.1 + 0.3 * exploration_signal + 0.3 * conflict_level - 0.2 * action_clarity
        primitive_probability = max(0.0, min(0.5, primitive_probability))

        return np.random.random() < primitive_probability

    def select_primitive(self, action_values):
        preferred_action = np.argmax(action_values)

        if preferred_action in [0, 3]:
            return 0
        else:
            return 1

    def process(self, action_values, conflict_level, exploration_signal):
        if self.current_primitive is not None:
            action = self.primitives[self.current_primitive][self.primitive_step]
            self.primitive_step += 1

            if self.primitive_step >= len(self.primitives[self.current_primitive]):
                self.current_primitive = None
                self.primitive_step = 0

            self.action_selection[action] += 1
            return action

        use_primitive = self.decide_primitive_use(action_values, conflict_level, exploration_signal)

        if use_primitive:
            self.current_primitive = self.select_primitive(action_values)
            self.primitive_step = 0
            action = self.primitives[self.current_primitive][self.primitive_step]
            self.primitive_step += 1
            self.primitive_usage_count += 1

            self.action_selection[action] += 1
            return action
        else:
            action = np.argmax(action_values)

            self.action_selection[action] += 1
            self.activation_count += 1
            return action