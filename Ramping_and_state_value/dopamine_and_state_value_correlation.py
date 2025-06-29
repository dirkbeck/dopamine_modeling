import gymnasium as gym
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
from scipy.stats import linregress
import tqdm
import matplotlib.colors as mcolors

np.random.seed(0)
random.seed(0)

class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=20):
        super().__init__()
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Discrete(grid_size*grid_size)
        self.action_space = gym.spaces.Discrete(4)
        self.max_timesteps = 10000
        self.start_state = (grid_size//2, grid_size//2)
        self.current_state = self.start_state
        self.timestep_count = 0
        self.action_to_direction = {
            0: (0,  1),   # right
            1: (0, -1),   # left
            2: (1,  0),   # down
            3: (-1, 0)    # up
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_state = self.start_state
        self.timestep_count = 0
        return self.current_state, {}

    def step(self, action):
        self.timestep_count += 1
        r, c = self.current_state
        dr, dc = self.action_to_direction[action]
        nr, nc = r+dr, c+dc
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size):
            nr, nc = r, c
            reward = -0.5
        else:
            reward = -0.05
            if (nr, nc) in food_patches and food_patches[(nr, nc)] > 0:
                reward += 10.0
                food_patches[(nr, nc)] -= 1
        self.current_state = (nr, nc)
        done = False
        truncated = (self.timestep_count >= self.max_timesteps)
        return self.current_state, reward, done, truncated, {}

class QLearningAgent:
    def __init__(self, action_space, alpha=0.2, gamma=0.95, beta=5.0):
        self.actions = list(range(action_space.n))
        self.q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def choose_action(self, state):
        qs = self.q[state].astype(float)
        maxq = np.max(qs)
        exp_q = np.exp(self.beta * (qs - maxq))
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.actions, p=probs)

    def update(self, s, a, r, s_next):
        pred = self.q[s][a]
        targ = r + self.gamma * np.max(self.q[s_next])
        self.q[s][a] += self.alpha * (targ - pred)

    def policy_entropy(self, state):
        qs = self.q[state].astype(float)
        maxq = np.max(qs)
        exp_q = np.exp(self.beta * (qs - maxq))
        p = exp_q / np.sum(exp_q)
        return -np.sum(p * np.log(p + 1e-9))

GRID_SIZE = 20
NUM_PATCHES = 5
INIT_FOOD = 25
MAX_STEPS = 10000
ROLLING_WIN = 100

random.seed(0)
initial_food_patches = {}
while len(initial_food_patches) < NUM_PATCHES:
    loc = (random.randrange(GRID_SIZE), random.randrange(GRID_SIZE))
    if loc != (GRID_SIZE//2, GRID_SIZE//2):
        initial_food_patches[loc] = INIT_FOOD

def run_foraging(max_steps, beta=5.0):
    global food_patches
    food_patches = initial_food_patches.copy()
    env = GridWorldEnv(GRID_SIZE)
    agent = QLearningAgent(env.action_space, alpha=0.2, gamma=0.95, beta=beta)

    traj, info_gain, state_val = [], [], []

    state, _ = env.reset()
    for step in tqdm.tqdm(range(max_steps)):
        a = agent.choose_action(state)
        next_state, r, done, trunc, _ = env.step(a)

        V = np.max(agent.q[state]) if state in agent.q else 0.0
        ent = agent.policy_entropy(state)
        info_gain.append(math.log(4) - ent)
        state_val.append(V)

        agent.update(state, a, r, next_state)
        traj.append(state)
        state = next_state
        if done or trunc:
            break

    return traj, info_gain, state_val, agent.q

def rolling_avg(x, w):
    x = np.asarray(x, float)
    if w<2 or x.size<2: return x
    return np.convolve(x, np.ones(w)/w, mode='same')

trajectory, info_gain, state_val, final_q = run_foraging(MAX_STEPS, beta=5.0)

roll_info = rolling_avg(info_gain, ROLLING_WIN)
roll_val = rolling_avg(state_val, ROLLING_WIN)
steps = np.arange(len(roll_info))

mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})

# figure 1 - value map and time series
fig1, axes1 = plt.subplots(1, 2, figsize=(6.0, 2.5))
fig1.subplots_adjust(wspace=0.4)

# value map
ax = axes1[0]
valmap = np.zeros((GRID_SIZE, GRID_SIZE))
all_vals = [np.max(v) for v in final_q.values()] if final_q else [0.0]
vmin, vmax = min(all_vals), max(all_vals)
for (r,c), qs in final_q.items():
    valmap[r,c] = np.max(qs) if len(qs)>0 else 0.0
cmap = mcolors.LinearSegmentedColormap.from_list('grey_red', ['#D3D3D3','red'])
im = ax.imshow(valmap, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
cbar = fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('State value')

ax.scatter(GRID_SIZE//2, GRID_SIZE//2, marker='^', c='lime', s=60,
           edgecolors='black', label='Start', zorder=5)
er, ec = trajectory[-1]
ax.scatter(ec, er, marker='s', c='red', s=60,
           edgecolors='black', label='End', zorder=5)
px = [c for r,c in initial_food_patches]
py = [r for r,c in initial_food_patches]
ax.scatter(px, py, marker='*', c='orange', s=80,
           edgecolors='black', label='Patches', zorder=4)
ax.set_title('Learned Value Map')
ax.set_xlabel('Column'); ax.set_ylabel('Row')
ax.set_xticks(np.linspace(0, GRID_SIZE-1, 5))
ax.set_yticks(np.linspace(0, GRID_SIZE-1, 5))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', fontsize=7, frameon=False)

# rolling averages
ax = axes1[1]
color1 = 'red'; color2 = 'blue'
ax.plot(steps, roll_info, color=color1, lw=1.2, label=f'Policy-IG (w={ROLLING_WIN})')
ax.set_xlabel('Step'); ax.set_ylabel('Policy-IG', color=color1)
ax.tick_params(axis='y', labelcolor=color1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax2 = ax.twinx()
ax2.plot(steps, roll_val, '--', color=color2, lw=1.2, label=f'State Value (w={ROLLING_WIN})')
ax2.set_ylabel('State Value', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.spines['top'].set_visible(False)

h1,l1 = ax.get_legend_handles_labels()
h2,l2 = ax2.get_legend_handles_labels()
ax2.legend(h1+h2, l1+l2, loc='upper right', fontsize=7, frameon=False)
ax.set_title('Rolling Policy-IG & State Value')

fig1.tight_layout(rect=[0,0.03,1,0.95])
fig1.savefig('figure1_value_and_policyIG.pdf', bbox_inches='tight')

# figure 2 - scatter plot with regression
s_all, i_all, r_all, p_all, _ = linregress(roll_info, roll_val)
xs = np.linspace(np.nanmin(roll_info), np.nanmax(roll_info), 200)

fig2, ax = plt.subplots(figsize=(4, 3))
sc = ax.scatter(roll_info, roll_val,
                c=np.arange(len(roll_info)),
                cmap='viridis', s=20, alpha=0.6)
ax.plot(xs, s_all*xs + i_all, 'k--', lw=1.5, label=f'R={r_all:.2f}')
cbar = fig2.colorbar(sc, ax=ax)
cbar.set_label('Time step')

ax.set_xlabel('Policy-IG')
ax.set_ylabel('State Value')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=7)

fig2.tight_layout()
fig2.savefig('figure2_time_colored_policyIG.pdf', bbox_inches='tight')
plt.show()