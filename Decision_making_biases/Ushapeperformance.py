import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.optimize import brentq
import matplotlib as mpl

random.seed(0)
np.random.seed(0)


def softmax(q_values, beta):
    q_stable = q_values - np.max(q_values)
    exp_q = np.exp(beta * q_stable)
    return exp_q / np.sum(exp_q)


def choose_action(probs):
    return np.random.choice(len(probs), p=probs)


def bandit_step_v3(action, current_episode, switch_episode):
    if current_episode < switch_episode:
        rewards = [0.7, 0.3]
    else:
        rewards = [0.45, 0.55]
    return 1.0 if random.random() < rewards[action] else 0.0


def bandit_step_v2(action, current_episode, switch_episode):
    if current_episode < switch_episode:
        rewards = [0.7, 0.3]
    else:
        rewards = [0.4, 0.6]
    return 1.0 if random.random() < rewards[action] else 0.0


def calculate_entropy(q_values, beta):
    if beta == 0:
        return np.log(len(q_values))
    probs = softmax(q_values, beta)
    return -np.sum(probs * np.log(probs + 1e-12))


def find_beta_for_entropy(target_entropy, q_values):
    if np.allclose(q_values, q_values[0]):
        return 0.0 if np.isclose(target_entropy, np.log(len(q_values))) else None

    def entropy_diff(beta):
        return calculate_entropy(q_values, beta) - target_entropy

    try:
        return brentq(entropy_diff, 1e-7, 1000.0)
    except:
        return None


def run_simulation(bandit_func, theoretical_q, task_name):
    print(f"Running {task_name}...")

    n_learning = 500
    n_eval = 100
    n_sims = 100
    alpha = 0.2
    switch_episode = 250

    max_entropy = np.log(2)
    target_entropies = np.linspace(0.05, max_entropy * 0.98, 7)

    results = []
    valid_entropies = []
    betas_used = []

    for target_entropy in target_entropies:
        beta = find_beta_for_entropy(target_entropy, theoretical_q)
        if beta is None:
            continue

        valid_entropies.append(target_entropy)
        betas_used.append(beta)

        sim_rewards = []
        for _ in range(n_sims):
            q_values = np.zeros(2)

            for episode in range(n_learning):
                probs = softmax(q_values, beta)
                action = choose_action(probs)
                reward = bandit_func(action, episode, switch_episode)
                q_values[action] += alpha * (reward - q_values[action])

            eval_reward = 0
            for _ in range(n_eval):
                probs = softmax(q_values, beta)
                action = choose_action(probs)
                reward = bandit_func(action, switch_episode + 1, switch_episode)
                eval_reward += reward

            sim_rewards.append(eval_reward / n_eval)

        results.append((np.mean(sim_rewards), stats.sem(sim_rewards)))

    return np.array(valid_entropies), np.array([r[0] for r in results]), np.array([r[1] for r in results]), np.array(
        betas_used)


# run simulations
entropies_1, rewards_1, sems_1, betas_1 = run_simulation(
    bandit_step_v3, np.array([0.45, 0.55]), "Task 1"
)

entropies_2, rewards_2, sems_2, betas_2 = run_simulation(
    bandit_step_v2, np.array([0.40, 0.60]), "Task 2"
)

# calculate policy information gain
max_entropy = np.log(2)
policy_ig_1 = max_entropy - entropies_1
policy_ig_2 = max_entropy - entropies_2

# find optimal points
opt_idx_1 = np.argmax(rewards_1)
opt_idx_2 = np.argmax(rewards_2)

plt.figure(figsize=(4, 4))

plt.errorbar(policy_ig_1, rewards_1, yerr=sems_1, marker='s', linestyle='--',
             label='Task 1', color='blue', capsize=3)
plt.axvline(policy_ig_1[opt_idx_1], color='blue', linestyle=':',
            label='Optimal policy-IG Task 1')
plt.plot(policy_ig_1[opt_idx_1], rewards_1[opt_idx_1], 'bo', markersize=8,
         markeredgecolor='black')

plt.errorbar(policy_ig_2, rewards_2, yerr=sems_2, marker='x', linestyle='-.',
             label='Task 2', color='red', capsize=3)
plt.axvline(policy_ig_2[opt_idx_2], color='red', linestyle=':',
            label='Optimal policy-IG Task 2')
plt.plot(policy_ig_2[opt_idx_2], rewards_2[opt_idx_2], 'rx', markersize=8,
         markeredgecolor='black')

plt.xlabel('Policy-IG (nats)')
plt.ylabel('Performance (average reward)')
plt.legend(fontsize='small', loc='lower center', bbox_to_anchor=(0.5, -0.5))
plt.tight_layout(rect=[0, 0.15, 1, 1])

mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('U_shape_PolicyIG.pdf', dpi=300, bbox_inches='tight')
plt.show()