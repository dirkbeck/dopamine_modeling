import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['xtick.major.width'] = 0.5
mpl.rcParams['ytick.major.width'] = 0.5
mpl.rcParams['xtick.minor.width'] = 0.3
mpl.rcParams['ytick.minor.width'] = 0.3
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

np.random.seed(0)


def run_sim(temp, teleport_t=None):
    x, y = 0, 0
    xg, yg = 100, 100
    d0 = abs(x - xg) + abs(y - yg)
    ig_trace = []
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for t in range(10000):
        d = abs(x - xg) + abs(y - yg)
        ig_trace.append(1.0 * (1 - d / d0))
        if (x, y) == (xg, yg):
            break

        if teleport_t and t == teleport_t:
            x = (x + xg) // 2
            y = (y + yg) // 2
            continue

        # boltzmann step
        d_news = [abs((x + dx) - xg) + abs((y + dy) - yg) for dx, dy in moves]
        prefs = - np.array(d_news) / temp
        prefs = prefs - np.max(prefs)
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / exp_prefs.sum()
        choice = np.random.choice(4, p=probs)
        dx, dy = moves[choice]
        x += dx
        y += dy

    if ig_trace and (x, y) == (xg, yg):
        ig_trace[-1] = 1.0
    return ig_trace


# figure 1 - different rewards
# using temperature as the reward values -- because the bigger the reward,
# the more value you get by going towards it, so the less random the policy is
temps = [1.0, 2.0, 5.0]
fig, ax = plt.subplots(figsize=(1.8, 1.4))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, tau in enumerate(temps):
    ig_trace = run_sim(tau)
    ax.plot(ig_trace, color=colors[i], linewidth=1, label=f'reward = {1 / tau:.1f}')

ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('policy-IG', fontsize=12)
ax.legend(fontsize=10, frameon=False, loc='upper right')
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('ramps.pdf', dpi=300, bbox_inches='tight')

# figure 2 - teleport
ig_tel = run_sim(1.0, teleport_t=100)
delta_ig = np.diff(ig_tel)

fig, ax = plt.subplots(figsize=(2.5, 1.8))
ax.plot(delta_ig, '-', color='#1f77b4', linewidth=1)
ax.axvline(100, linestyle='--', color='gray', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Time step', fontsize=12)
ax.set_ylabel('Δ policy-IG', fontsize=12)
ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig('teleport.pdf', dpi=300, bbox_inches='tight')
plt.show()