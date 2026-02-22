import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# softmax function
def softmax(q, tau=1.0):
    q_stable = q - np.max(q)
    exp_q = np.exp(q_stable / tau)
    return exp_q / np.sum(exp_q)

# entropy calc
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p)) if len(p) > 0 else 0.0

# params
alpha = 0.1
gamma = 0.9
tau = 1.0
n_episodes = 500

# set up plotting style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none'
})

# envs to test
env_names = ["Taxi-v3", "CliffWalking-v1", "FrozenLake-v1"]

def run_env(env_name):
    env = gym.make(env_name, is_slippery=True) if "FrozenLake" in env_name else gym.make(env_name)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    rpe_list, policy_ig = [], []

    for ep in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            probs = softmax(q_table[state], tau=tau)
            action = np.random.choice(len(probs), p=probs)
            h_before = entropy(probs)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            target = 0 if done else np.max(q_table[next_state])
            rpe = reward + gamma * target - q_table[state, action]
            rpe_list.append(rpe)

            q_table[state, action] += alpha * rpe

            probs_after = softmax(q_table[state], tau=tau)
            h_after = entropy(probs_after)
            policy_ig.append(h_before - h_after)
            state = next_state

    env.close()
    return {"rpe": np.array(rpe_list), "policy_ig": np.array(policy_ig)}

all_data = {}
for env_name in env_names:
    short = env_name.split('-')[0].lower()
    all_data[short] = run_env(env_name)

fig, axes = plt.subplots(1, 3, figsize=(3.5, 1))

# taxi plot
data = all_data['taxi']
axes[0].scatter(data["rpe"], data["policy_ig"], s=10, alpha=.5, color='k', edgecolors='none', rasterized=True)
axes[0].set_title('taxi', fontsize=7)
axes[0].set_xlabel("RPE", fontsize=6)
axes[0].set_ylabel("Δ Policy-IG (nats)", fontsize=6)
axes[0].axhline(0, color='grey', lw=0.5, alpha=1.0)
axes[0].axvline(0, color='grey', lw=0.5, alpha=1.0)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].tick_params(labelsize=6)

# cliff plot
data = all_data['cliffwalking']
axes[1].scatter(data["rpe"], data["policy_ig"], s=10, alpha=.5, color='k', edgecolors='none', rasterized=True)
axes[1].set_title('cliffwalking', fontsize=7)
axes[1].set_xlabel("RPE", fontsize=6)
axes[1].axhline(0, color='grey', lw=0.5, alpha=1.0)
axes[1].axvline(0, color='grey', lw=0.5, alpha=1.0)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].tick_params(labelsize=6)

# frozen lake plot
data = all_data['frozenlake']
axes[2].scatter(data["rpe"], data["policy_ig"], s=10, alpha=.5, color='k', edgecolors='none', rasterized=True)
axes[2].set_title('frozenlake', fontsize=7)
axes[2].set_xlabel("RPE", fontsize=6)
axes[2].axhline(0, color='grey', lw=0.5, alpha=1.0)
axes[2].axvline(0, color='grey', lw=0.5, alpha=1.0)
axes[2].spines['top'].set_visible(False)
axes[2].spines['right'].set_visible(False)
axes[2].tick_params(labelsize=6)

plt.tight_layout()
plt.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.20, wspace=0.15)

plt.savefig("open_gym_comparison.pdf", dpi=300, bbox_inches='tight', pad_inches=0.02,
            transparent=False, facecolor='white', edgecolor='none')
plt.show()