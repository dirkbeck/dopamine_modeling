import gymnasium as gym
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import tqdm
import matplotlib as mpl

np.random.seed(0)
random.seed(0)


class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=4):
        super(GridWorldEnv, self).__init__()
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Discrete(grid_size * grid_size)
        self.action_space = gym.spaces.Discrete(4)
        self.max_timesteps = 100
        self.goal_state = grid_size * grid_size - 1
        self.start_state = 0
        self.current_state = self.start_state
        self.timestep_count = 0
        self.action_to_direction = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.timestep_count = 0
        return self.current_state, {}

    def step(self, action):
        self.timestep_count += 1
        row, col = divmod(self.current_state, self.grid_size)
        dr, dc = self.action_to_direction[action]
        new_row, new_col = row + dr, col + dc
        new_row = max(0, min(new_row, self.grid_size - 1))
        new_col = max(0, min(new_col, self.grid_size - 1))
        new_state = new_row * self.grid_size + new_col
        self.current_state = new_state

        if new_state == self.goal_state:
            reward = 2
            terminated = True
        else:
            reward = -1
            terminated = False

        truncated = self.timestep_count >= self.max_timesteps
        return new_state, reward, terminated, truncated, {}


class QLearningAgent:
    def __init__(self, obs_space, action_space, lr=0.1, gamma=0.9, tau=1.0):
        self.obs_space = obs_space
        self.action_space = action_space
        self.lr = lr
        self.gamma = gamma
        self.tau = tau                   # temperature for softmax
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))

    def choose_action(self, state):
        q_values = self.q_table[state]
        # subtract max for numerical stability
        z = q_values / self.tau
        z = z - np.max(z)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z)
        # sample according to softmax probabilities
        return np.random.choice(self.action_space.n, p=probs)

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - predict)

    def calc_entropy(self, state):
        q_values = self.q_table[state]
        z = q_values / self.tau
        z = z - np.max(z)
        exp_z = np.exp(z)
        probs = exp_z / np.sum(exp_z)
        return -np.sum(probs * np.log(probs + 1e-9))


# params (arbitrary)
num_episodes = 100
n_sim = 10
lr = 0.1
gamma = 0.9
tau = 1
grid_size = 4
surprise_prob = 0.05
surprise_mags = np.linspace(-20, 0, 10)

all_means = []
all_stds = []

for surprise_mag in surprise_mags:
    delta_entropies = []

    for sim in tqdm.tqdm(range(n_sim)):
        env = GridWorldEnv(grid_size=grid_size)
        agent = QLearningAgent(env.observation_space, env.action_space, lr, gamma, tau)
        entropies = []
        surprise_eps = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            surprise_happened = False

            while not done and not truncated:
                action = agent.choose_action(state)
                next_state, reward, done, truncated, _ = env.step(action)

                # add surprise in second half
                if episode > num_episodes // 2 and random.random() < surprise_prob:
                    reward += surprise_mag
                    surprise_happened = True

                agent.learn(state, action, reward, next_state)
                state = next_state

            # calculate total entropy for this episode
            total_entropy = 0
            for s in range(env.observation_space.n):
                total_entropy += agent.calc_entropy(s)
            entropies.append(total_entropy)

            if surprise_happened:
                surprise_eps.append(episode)

        # get entropy changes
        delta_entropy = np.diff(entropies)
        avg_delta = 0.0

        if surprise_eps:
            deltas = [delta_entropy[i - 1] for i in surprise_eps[1:]
                      if i > 0 and i < len(delta_entropy)]
            if deltas:
                avg_delta = np.mean(deltas)

        delta_entropies.append(avg_delta)

    all_means.append(np.mean(delta_entropies))
    all_stds.append(np.std(delta_entropies))

# plotting
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 6,
    'axes.linewidth': 0.8,
    'axes.labelsize': 6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(2.5, 1))

# flip sign on delta-entropy to get policy-IG
means = [-m for m in all_means]

ax.errorbar(surprise_mags, means, yerr=all_stds,
            marker='.', capsize=2, color='black',
            linestyle='-', elinewidth=1, markeredgewidth=0.8)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.set_xlabel('RPE', fontsize=6)
ax.set_ylabel('Δ Policy-IG (nats)', fontsize=6)
ax.axhline(0, color='black', linewidth=0.8)

min_idx = np.argmin(np.abs(means))
ax.axvline(surprise_mags[min_idx], color='gray',
           linestyle='--', linewidth=0.8)

fig.tight_layout()
fig.savefig('foraging_with_large_costs.pdf', bbox_inches='tight')
plt.show()