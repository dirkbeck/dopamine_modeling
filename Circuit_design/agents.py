import random
import numpy as np
import layers
from tqdm import tqdm


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def kl_divergence(p, q):
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))


class Agent:
    def __init__(self, env, name='Agent', use_direct_env_input=False,
                 has_critic=True, has_governor=True, fixed_temperature=None):
        self.env = env
        self.name = name
        self.use_direct_env_input = use_direct_env_input
        self.has_critic = has_critic
        self.has_governor = has_governor
        self.fixed_temperature = fixed_temperature

        self.actions = [0, 1, 2, 3]
        self.num_actions = len(self.actions)
        self.Q = np.zeros((25, len(self.actions)))
        self.H = np.ones(25) * 1.0
        self.alpha = 0.1
        self.gamma = 0.9

        self.env_stimuli = layers.EnvironmentalStimuliLayer(grid_size=env.grid_size)
        self.short_term = layers.ShortTermPlanningLayer(num_actions=len(self.actions))
        self.long_term = layers.LongTermPlanningLayer(num_actions=len(self.actions))
        self.conflict = layers.ConflictResolutionLayer(num_actions=len(self.actions))
        self.motor_program = layers.MotorProgramsLayer(num_actions=len(self.actions))

        self.layer_impacts = {layer: {'actor': [], 'critic': [], 'governor': []} for layer in
                              ['env_stimuli', 'short_term', 'long_term', 'conflict', 'motor']}
        self.layer_impacts_over_time = {layer: {'actor': [], 'critic': [], 'governor': []} for layer in
                                        ['env_stimuli', 'short_term', 'long_term', 'conflict', 'motor']}
        self.cached_states = []
        self.cached_features = []
        self.cached_actions = []
        self.episode_sensitivities = {}
        self.component_usage = {"actor": 0, "critic": 0, "governor": 0}
        self.region_component_usage = {0: {'actor': [], 'critic': [], 'governor': []},
                                       1: {'actor': [], 'critic': [], 'governor': []}}
        self.current_region = 0
        self.steps_in_region = {0: 0, 1: 0}
        self.episode_region_usage = {0: {'actor': 0, 'critic': 0, 'governor': 0},
                                     1: {'actor': 0, 'critic': 0, 'governor': 0}}
        self.st_predictions = {}
        self.lt_predictions = {}

    def reset(self):
        if not self.use_direct_env_input:
            if hasattr(self.env_stimuli, 'reset'):
                self.env_stimuli.reset()
            if hasattr(self.short_term, 'reset'):
                self.short_term.reset()

        self.st_predictions.clear()
        self.lt_predictions.clear()
        self.episode_region_usage = {0: {'actor': 0, 'critic': 0, 'governor': 0},
                                     1: {'actor': 0, 'critic': 0, 'governor': 0}}
        self.steps_in_region = {0: 0, 1: 0}
        self.episode_sensitivities = {layer: {'actor': 0, 'critic': 0, 'governor': 0} for layer in self.layer_impacts}
        self.cached_states.clear()
        self.cached_features.clear()
        self.cached_actions.clear()

    def perform_sensitivity_analysis(self, state, features, state_idx):
        if self.use_direct_env_input or len(self.cached_states) >= 100:
            return

        self.cached_states.append(state)
        self.cached_features.append(features)

        stp_values = self.short_term.process(features, state_idx)
        ltp_values, ltp_uncertainty = self.long_term.plan(state_idx)
        combined_values, conflict_level, combined_uncertainty = self.conflict.process(
            stp_values, ltp_values, ltp_uncertainty)

        baseline_probs = softmax(combined_values)
        baseline_action = np.argmax(baseline_probs)
        baseline_q = self.Q[state_idx, baseline_action]
        baseline_h = self.H[state_idx]

        neutral_features = np.zeros_like(features)
        neutral_state_idx = 0

        env_stp_values = self.short_term.process(neutral_features, neutral_state_idx)
        env_ltp_values, env_ltp_uncertainty = self.long_term.plan(neutral_state_idx)
        env_combined_values, _, env_combined_uncertainty = self.conflict.process(
            env_stp_values, env_ltp_values, env_ltp_uncertainty)
        env_probs = softmax(env_combined_values)
        env_actor_impact = kl_divergence(baseline_probs, env_probs)
        env_critic_impact = abs(baseline_q - self.Q[neutral_state_idx, baseline_action]) / (baseline_q + 1e-10)
        env_governor_impact = abs(baseline_h - self.H[neutral_state_idx]) / (baseline_h + 1e-10)

        st_stp_values = np.zeros_like(stp_values)
        st_combined_values, _, _ = self.conflict.process(
            st_stp_values, ltp_values, ltp_uncertainty)
        st_probs = softmax(st_combined_values)
        st_actor_impact = kl_divergence(baseline_probs, st_probs)
        st_critic_impact = abs(np.mean(self.short_term.td_errors)) if self.short_term.td_errors else 0
        st_governor_impact = abs(combined_uncertainty - ltp_uncertainty) / (combined_uncertainty + 1e-10)

        lt_ltp_values = np.zeros_like(ltp_values)
        lt_ltp_uncertainty = 0.5
        lt_combined_values, _, _ = self.conflict.process(
            stp_values, lt_ltp_values, lt_ltp_uncertainty)
        lt_probs = softmax(lt_combined_values)
        lt_actor_impact = kl_divergence(baseline_probs, lt_probs)
        lt_critic_impact = abs(np.mean(self.long_term.value_changes)) if self.long_term.value_changes else 0
        lt_governor_impact = abs(ltp_uncertainty - 0.5) / (ltp_uncertainty + 1e-10)

        cf_combined_values = (stp_values + ltp_values) / 2.0
        cf_probs = softmax(cf_combined_values)
        cf_actor_impact = kl_divergence(baseline_probs, cf_probs)
        cf_critic_impact = 0.1
        cf_governor_impact = abs(combined_uncertainty - 0.5) / (combined_uncertainty + 1e-10)

        motor_action = np.argmax(combined_values)
        motor_baseline_action = self.motor_program.process(combined_values, conflict_level, combined_uncertainty)
        motor_actor_impact = 1.0 if motor_action != motor_baseline_action else 0.0
        motor_critic_impact = abs(self.Q[state_idx, motor_action] - self.Q[state_idx, motor_baseline_action]) / (
                    baseline_q + 1e-10)
        motor_governor_impact = 0.1

        self.layer_impacts['env_stimuli']['actor'].append(env_actor_impact)
        self.layer_impacts['env_stimuli']['critic'].append(env_critic_impact)
        self.layer_impacts['env_stimuli']['governor'].append(env_governor_impact)
        self.layer_impacts['short_term']['actor'].append(st_actor_impact)
        self.layer_impacts['short_term']['critic'].append(st_critic_impact)
        self.layer_impacts['short_term']['governor'].append(st_governor_impact)
        self.layer_impacts['long_term']['actor'].append(lt_actor_impact)
        self.layer_impacts['long_term']['critic'].append(lt_critic_impact)
        self.layer_impacts['long_term']['governor'].append(lt_governor_impact)
        self.layer_impacts['conflict']['actor'].append(cf_actor_impact)
        self.layer_impacts['conflict']['critic'].append(cf_critic_impact)
        self.layer_impacts['conflict']['governor'].append(cf_governor_impact)
        self.layer_impacts['motor']['actor'].append(motor_actor_impact)
        self.layer_impacts['motor']['critic'].append(motor_critic_impact)
        self.layer_impacts['motor']['governor'].append(motor_governor_impact)

        self.episode_sensitivities['env_stimuli']['actor'] += env_actor_impact
        self.episode_sensitivities['env_stimuli']['critic'] += env_critic_impact
        self.episode_sensitivities['env_stimuli']['governor'] += env_governor_impact
        self.episode_sensitivities['short_term']['actor'] += st_actor_impact
        self.episode_sensitivities['short_term']['critic'] += st_critic_impact
        self.episode_sensitivities['short_term']['governor'] += st_governor_impact
        self.episode_sensitivities['long_term']['actor'] += lt_actor_impact
        self.episode_sensitivities['long_term']['critic'] += lt_critic_impact
        self.episode_sensitivities['long_term']['governor'] += lt_governor_impact
        self.episode_sensitivities['conflict']['actor'] += cf_actor_impact
        self.episode_sensitivities['conflict']['critic'] += cf_critic_impact
        self.episode_sensitivities['conflict']['governor'] += cf_governor_impact
        self.episode_sensitivities['motor']['actor'] += motor_actor_impact
        self.episode_sensitivities['motor']['critic'] += motor_critic_impact
        self.episode_sensitivities['motor']['governor'] += motor_governor_impact

    def get_state_idx_direct(self, state):
        row, col = state
        row_idx = min(self.env.grid_size - 1, max(0, int(row)))
        col_idx = min(self.env.grid_size - 1, max(0, int(col)))
        return row_idx * self.env.grid_size + col_idx

    def policy(self, state):
        self.current_region = self.env.get_region(state)
        self.steps_in_region[self.current_region] += 1

        pre_actor = self.component_usage.get("actor", 0)
        pre_governor = self.component_usage.get("governor", 0)

        action_values = None
        action = None
        features = None
        state_idx = -1

        if self.use_direct_env_input:
            state_idx = self.get_state_idx_direct(state)
            q_vals = self.Q[state_idx]

            if self.has_governor:
                temperature = self.H[state_idx]
                self.component_usage["governor"] += 1
            elif self.fixed_temperature is not None:
                temperature = self.fixed_temperature
            else:
                temperature = 1.0

            beta = 1.0
            safe_temp = max(temperature, 1e-5)
            exp_vals = np.exp(beta * q_vals / safe_temp)
            probs = softmax(exp_vals)

            action = np.random.choice(self.actions, p=probs)
            self.component_usage["actor"] += 1

        else:
            features = self.env_stimuli.process(state)
            row_pos, col_pos = features[0:2]
            row_idx = min(self.env.grid_size - 1, max(0, int(row_pos * self.env.grid_size)))
            col_idx = min(self.env.grid_size - 1, max(0, int(col_pos * self.env.grid_size)))
            state_idx = row_idx * self.env.grid_size + col_idx

            stp_values = self.short_term.process(features, state_idx)
            self.st_predictions[state_idx] = stp_values.copy()

            ltp_values, ltp_uncertainty = self.long_term.plan(state_idx)
            self.lt_predictions[state_idx] = ltp_values.copy()

            combined_values, conflict_level, combined_uncertainty = self.conflict.process(
                stp_values, ltp_values, ltp_uncertainty)

            if self.has_governor:
                temperature = self.H[state_idx]
                self.component_usage["governor"] += 1
            elif self.fixed_temperature is not None:
                temperature = self.fixed_temperature
            else:
                temperature = 1.0

            beta = 1.0
            safe_temp = max(temperature, 1e-5)
            action = self.motor_program.process(combined_values, conflict_level, combined_uncertainty)

            self.component_usage["actor"] += 1

            self.perform_sensitivity_analysis(state, features, state_idx)

        actor_delta = self.component_usage.get("actor", 0) - pre_actor
        governor_delta = self.component_usage.get("governor", 0) - pre_governor
        self.episode_region_usage[self.current_region]['actor'] += actor_delta
        if self.has_governor:
            self.episode_region_usage[self.current_region]['governor'] += governor_delta

        if not self.use_direct_env_input:
            self.cached_actions.append(action)

        return action

    def update(self, state, action, reward, next_state):
        region = self.env.get_region(state)

        pre_critic = self.component_usage.get("critic", 0)
        pre_governor = self.component_usage.get("governor", 0)

        state_idx = -1
        next_state_idx = -1
        features = None

        if self.use_direct_env_input:
            state_idx = self.get_state_idx_direct(state)
            next_state_idx = self.get_state_idx_direct(next_state)
        else:
            features = self.env_stimuli.extract_features(state)
            row_pos, col_pos = features[0:2]
            row_idx = min(self.env.grid_size - 1, max(0, int(row_pos * self.env.grid_size)))
            col_idx = min(self.env.grid_size - 1, max(0, int(col_pos * self.env.grid_size)))
            state_idx = row_idx * self.env.grid_size + col_idx

            next_features = self.env_stimuli.extract_features(next_state)
            next_row_pos, next_col_pos = next_features[0:2]
            next_row_idx = min(self.env.grid_size - 1, max(0, int(next_row_pos * self.env.grid_size)))
            next_col_idx = min(self.env.grid_size - 1, max(0, int(next_col_pos * self.env.grid_size)))
            next_state_idx = next_row_idx * self.env.grid_size + next_col_idx

        td_error = 0
        if state_idx != -1 and next_state_idx != -1:
            best_next = np.max(self.Q[next_state_idx])
            td_error = reward + self.gamma * best_next - self.Q[state_idx, action]

            if self.has_critic:
                self.Q[state_idx, action] += self.alpha * td_error
                self.component_usage["critic"] += 1

            if self.has_governor:
                self.H[state_idx] += 0.01 * abs(td_error)
                self.H[state_idx] = max(0.5, self.H[state_idx] * 0.99)
                self.component_usage["governor"] += 1

        if not self.use_direct_env_input and state_idx != -1 and next_state_idx != -1:
            self.short_term.remember(state_idx, action, reward)
            self.long_term.update_model(state_idx, action, next_state_idx, reward)

            if state_idx in self.st_predictions and state_idx in self.lt_predictions:
                target_value = reward + self.gamma * best_next
                st_pred_value = self.st_predictions[state_idx][action]
                lt_pred_value = self.lt_predictions[state_idx][action]

                st_error = abs(st_pred_value - target_value)
                lt_error = abs(lt_pred_value - target_value)

                if st_error + lt_error > 1e-5:
                    self.conflict.update_confidence(st_error, lt_error)
                else:
                    self.conflict.st_confidence = 0.5
                    self.conflict.lt_confidence = 0.5

        critic_delta = self.component_usage.get("critic", 0) - pre_critic
        governor_delta = self.component_usage.get("governor", 0) - pre_governor
        if self.has_critic:
            self.episode_region_usage[region]['critic'] += critic_delta
        if self.has_governor:
            self.episode_region_usage[region]['governor'] += governor_delta

    def finalize_episode(self):
        for region in [0, 1]:
            steps = self.steps_in_region[region]
            if steps > 0:
                norm_actor = self.episode_region_usage[region]['actor'] / steps
                norm_critic = self.episode_region_usage[region]['critic'] / steps if self.has_critic else 0
                norm_governor = self.episode_region_usage[region]['governor'] / steps if self.has_governor else 0
                self.region_component_usage[region]['actor'].append(norm_actor)
                self.region_component_usage[region]['critic'].append(norm_critic)
                self.region_component_usage[region]['governor'].append(norm_governor)
            else:
                self.region_component_usage[region]['actor'].append(0)
                self.region_component_usage[region]['critic'].append(0)
                self.region_component_usage[region]['governor'].append(0)

        if not self.use_direct_env_input and self.cached_states:
            num_sensitivity_steps = len(self.cached_states)
            for layer in self.episode_sensitivities:
                for component in self.episode_sensitivities[layer]:
                    normalized_value = self.episode_sensitivities[layer][component] / num_sensitivity_steps
                    self.layer_impacts_over_time[layer][component].append(normalized_value)
            self.cached_states.clear()
            self.cached_features.clear()
            self.cached_actions.clear()
        else:
            for layer in self.layer_impacts_over_time:
                for component in self.layer_impacts_over_time[layer]:
                    self.layer_impacts_over_time[layer][component].append(0)

    def learn(self, episodes=100):
        total_rewards = []
        desc_suffix = " (Direct)" if self.use_direct_env_input else " (Layered)"
        for ep in tqdm(range(episodes), desc=f"Training {self.name}{desc_suffix}", leave=False):
            state = self.env.reset()
            self.reset()

            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 200:
                action = self.policy(state)
                next_state, reward, done = self.env.step(action)

                self.update(state, action, reward, next_state)

                state = next_state
                episode_reward += reward
                step_count += 1

            self.finalize_episode()
            total_rewards.append(episode_reward)

        return total_rewards

    def get_layer_impacts(self):
        impact_metrics = {}
        if self.use_direct_env_input or not any(any(c for c in comp.values()) for comp in self.layer_impacts.values()):
            for layer in self.layer_impacts:
                impact_metrics[layer] = {'actor': 0.0, 'critic': 0.0, 'governor': 0.0}
        else:
            for layer, components in self.layer_impacts.items():
                impact_metrics[layer] = {}
                for component, impacts in components.items():
                    if impacts:
                        clean_impacts = [i for i in impacts if np.isfinite(i)]
                        if clean_impacts:
                            impact_metrics[layer][component] = np.mean(clean_impacts)
                        else:
                            impact_metrics[layer][component] = 0.0
                    else:
                        impact_metrics[layer][component] = 0.0

        return {
            "component_usage": self.component_usage,
            "layer_impacts": impact_metrics,
            "layer_impacts_over_time": self.layer_impacts_over_time
        }

    def get_region_component_data(self):
        return self.region_component_usage


class ActorCriticGovernorAgent(Agent):
    def __init__(self, env, use_direct_env_input=False):
        super().__init__(env, name='ACG', use_direct_env_input=use_direct_env_input,
                         has_critic=True, has_governor=True)


class ActorCriticAgent(Agent):
    def __init__(self, env, use_direct_env_input=False):
        super().__init__(env, name='AC', use_direct_env_input=use_direct_env_input,
                         has_critic=True, has_governor=False, fixed_temperature=1.0)


class ActorGovernorAgent(Agent):
    def __init__(self, env, use_direct_env_input=False):
        super().__init__(env, name='AG', use_direct_env_input=use_direct_env_input,
                         has_critic=False, has_governor=True)


class RandomAgent(Agent):
    def __init__(self, env):
        super().__init__(env, name='Random', use_direct_env_input=True,
                         has_critic=False, has_governor=False)

    def policy(self, state):
        self.current_region = self.env.get_region(state)
        self.steps_in_region[self.current_region] += 1
        pre_actor = self.component_usage.get("actor", 0)

        action = random.choice(self.actions)
        self.component_usage["actor"] += 1

        actor_delta = self.component_usage.get("actor", 0) - pre_actor
        self.episode_region_usage[self.current_region]['actor'] += actor_delta

        return action

    def update(self, state, action, reward, next_state):
        pass

    def finalize_episode(self):
        for region in [0, 1]:
            steps = self.steps_in_region[region]
            if steps > 0:
                norm_actor = self.episode_region_usage[region]['actor'] / steps
                self.region_component_usage[region]['actor'].append(norm_actor)
                self.region_component_usage[region]['critic'].append(0)
                self.region_component_usage[region]['governor'].append(0)
            else:
                self.region_component_usage[region]['actor'].append(0)
                self.region_component_usage[region]['critic'].append(0)
                self.region_component_usage[region]['governor'].append(0)