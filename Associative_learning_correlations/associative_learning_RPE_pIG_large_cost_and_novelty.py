import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

# try to get LOWESS
try:
    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    HAVE_LOWESS = True
except ImportError:
    print("skipping lowess")
    HAVE_LOWESS = False

epsilon = 1e-9

def safe_logsumexp(x):
    x = np.asarray(x)
    if x.size == 0: return -np.inf
    m = np.max(x)
    if not np.isfinite(m):
        if np.any(x == np.inf):
            return np.inf
        else:
            return -np.inf
    s = np.sum(np.exp(x - m))
    if s <= 0: return -np.inf
    return m + np.log(s)

def softmax_3d(q, beta):
    q = np.asarray(q, float)
    m = np.nanmax(beta * q)
    if not np.isfinite(m): m = 0
    expq = np.exp(np.clip(beta * q - m, -700, 700))
    S = np.nansum(expq)
    if S <= epsilon or not np.isfinite(S):
        return np.ones_like(q) / len(q)
    return expq / S

def calculate_entropy(p):
    p = np.clip(p, epsilon, 1.0)
    p = p / np.nansum(p)
    nz = p > epsilon
    return -np.nansum(p[nz] * np.log(p[nz])) if np.any(nz) else 0.0

def calculate_delta_info_gain(qb, qa, beta):
    p0 = softmax_3d(qb, beta)
    p1 = softmax_3d(qa, beta)
    H0 = calculate_entropy(p0)
    H1 = calculate_entropy(p1)
    return -(H1 - H0)

def run_sim(q_start, lr, beta, base_R, base_P, num_exp, episode_len, outcome_range, outcome_type='cost'):
    Q0 = q_start.copy()
    mn, mx = outcome_range
    cue = 'Cue'
    qb = Q0['Baseline']
    all_rpe, all_di = [], []
    possible = [0.0, base_R, mn, mx]
    clip_min, clip_max = min(possible) - 1, max(possible) + 1

    for exp in range(num_exp):
        Q = Q0.copy()
        for t in range(episode_len):
            qv = Q[cue]
            vb = np.array([qv, qb])
            R = base_R if random.random() < base_P else 0.0
            rpe = R - qv
            Q[cue] = np.clip(qv + lr * rpe, clip_min, clip_max)
            va = np.array([Q[cue], qb])
            di = calculate_delta_info_gain(vb, va, beta)
            all_rpe.append(rpe)
            all_di.append(di)

        # final trial with random outcome
        qv = Q[cue]
        vb = np.array([qv, qb])
        R = random.uniform(mn, mx)
        rpe = R - qv
        Q[cue] = np.clip(qv + lr * rpe, clip_min, clip_max)
        va = np.array([Q[cue], qb])
        di = calculate_delta_info_gain(vb, va, beta)
        all_rpe.append(rpe)
        all_di.append(di)

    return np.array(all_rpe), np.array(all_di)

def run_novelty_sim(q_start, lr, beta, base_R, base_P, num_exp, episode_len, final_range, novelty_mag):
    Q0 = q_start.copy()
    mn, mx = final_range
    cue = 'Cue'
    qb = Q0['Baseline']
    all_rpe, all_di = [], []
    novelty_rpe = np.nan
    novelty_di = np.nan
    possible = [0.0, base_R, mn, mx]
    clip_min = min(possible) - 1
    clip_max = max(possible) + abs(lr * novelty_mag) + 1

    for exp in range(num_exp):
        Q = Q0.copy()
        for t in range(episode_len):
            qv = Q[cue]
            vb = np.array([qv, qb])
            R = base_R if random.random() < base_P else 0.0
            rpe = R - qv
            Q[cue] = np.clip(qv + lr * rpe, clip_min, clip_max)
            va = np.array([Q[cue], qb])
            di = calculate_delta_info_gain(vb, va, beta)
            all_rpe.append(rpe)
            all_di.append(di)

        # final trial
        qv = Q[cue]
        vb = np.array([qv, qb])
        if exp == num_exp - 1:  # last experiment gets novelty
            Rexp = base_R * base_P
            rpe = Rexp - qv
            dq = lr * rpe + lr * novelty_mag
            Q[cue] = np.clip(qv + dq, clip_min, clip_max)
            va = np.array([Q[cue], qb])
            di = calculate_delta_info_gain(vb, va, beta)
            novelty_rpe = rpe
            novelty_di = di
        else:
            R = random.uniform(mn, mx)
            rpe = R - qv
            Q[cue] = np.clip(qv + lr * rpe, clip_min, clip_max)
            va = np.array([Q[cue], qb])
            di = calculate_delta_info_gain(vb, va, beta)
            all_rpe.append(rpe)
            all_di.append(di)

    return np.array(all_rpe), np.array(all_di), novelty_rpe, novelty_di

# Set params and run
random.seed(5)
np.random.seed(115)

lr = 0.15
beta = 3.0
q_start = {'Cue': 0.0, 'Baseline': 0.0}
base_R = 1.0
base_P = 0.80
episode_len = 20

lc_rpe, lc_di = run_sim(q_start, lr, beta, base_R, base_P, 20, episode_len, (-30.0, -5.0), 'cost')
mc_rpe, mc_di = run_sim(q_start, lr, beta, base_R, base_P, 30, episode_len, (-5.0, -0.1), 'cost')
lr_rpe, lr_di = run_sim(q_start, lr, beta, base_R, base_P, 20, episode_len, (5.0, 30.0), 'reward')
mr_rpe, mr_di = run_sim(q_start, lr, beta, base_R, base_P, 30, episode_len, (1.1, 5.0), 'reward')
nov_rpe, nov_di, nov_pt_rpe, nov_pt_di = run_novelty_sim(
    q_start, lr, beta, base_R, base_P, 50, episode_len, (1.1, 5.0), 15.0
)

cost_rpe = np.concatenate([lc_rpe, mc_rpe])
cost_di  = np.concatenate([lc_di, mc_di])
reward_rpe = np.concatenate([lr_rpe, mr_rpe])
reward_di  = np.concatenate([lr_di, mr_di])

# LOWESS smoothing if available
def get_smooth(x, y):
    if not HAVE_LOWESS:
        return np.array([]), np.array([])
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.array([]), np.array([])
    sm = lowess(y[m], x[m], frac=0.3, it=0)
    return sm[:, 0], sm[:, 1]

cost_xs,   cost_ys   = get_smooth(cost_rpe,   cost_di)
reward_xs, reward_ys = get_smooth(reward_rpe, reward_di)
nov_xs,    nov_ys    = get_smooth(nov_rpe,    nov_di)

# axis‐limit helpers
def get_ylim(data_arrays, percentiles=(0.5, 100), pad=0.1):
    all_data = np.concatenate([np.ravel(d) for d in data_arrays if len(d) > 0])
    all_data = all_data[np.isfinite(all_data)]
    if len(all_data) == 0:
        return (-0.2, 0.4)
    low, high = np.percentile(all_data, percentiles)
    rng = high - low
    padding = max(abs(rng) * pad, 0.05)
    return (low - padding, high + padding)

def get_xlim(data_arrays, pad=0.1):
    all_data = np.concatenate([np.ravel(d) for d in data_arrays if len(d) > 0])
    all_data = all_data[np.isfinite(all_data)]
    if len(all_data) == 0:
        return (-2, 2)
    low, high = all_data.min(), all_data.max()
    rng = high - low
    padding = max(abs(rng) * pad, 0.05)
    return (low - padding, high + padding)

# compute per‐panel y‐limits
cost_ylim    = get_ylim([cost_di])
reward_ylim  = get_ylim([reward_di])
novelty_ylim = get_ylim([nov_di, [nov_pt_di]])
y_lims = [cost_ylim, reward_ylim, novelty_ylim]

cost_xlim   = get_xlim([cost_rpe])
reward_xlim = get_xlim([reward_rpe])
nov_xlim    = get_xlim([nov_rpe, [nov_pt_rpe]])

# plotting
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

fig, axes = plt.subplots(3, 1, figsize=(2.0, 4.5))
fig.subplots_adjust(hspace=0.35)

titles = ["Episodic Costs", "Episodic Rewards", "Episodic Novelty"]
data_sets = [
    (cost_rpe,   cost_di,   cost_xs,   cost_ys,   cost_xlim),
    (reward_rpe, reward_di, reward_xs, reward_ys, reward_xlim),
    (nov_rpe,    nov_di,    nov_xs,    nov_ys,    nov_xlim)
]

for i, (ax, (x, y, xs, ys, xlim), title) in enumerate(zip(axes, data_sets, titles)):
    ax.set_title(title, fontsize=9)
    ax.set_ylim(y_lims[i])
    ax.set_xlim(xlim)
    ax.axhline(0, c='grey', lw=0.5)
    ax.axvline(0, c='grey', lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # scatter
    m = np.isfinite(x) & np.isfinite(y)
    if np.any(m):
        ax.scatter(x[m], y[m], color='k', s=20, alpha=0.3, edgecolors='none')

    # LOWESS curve
    if len(xs) > 0 and len(ys) > 0:
        m2 = np.isfinite(xs) & np.isfinite(ys)
        ax.plot(xs[m2], ys[m2], color='k', lw=1.2, alpha=0.9)

    ax.set_ylabel("$\\Delta$ Policy-IG\n(nats)", fontsize=8)
    ax.set_xlabel("RPE", fontsize=9)

    # novelty star on last panel
    if i == 2 and np.isfinite(nov_pt_rpe):
        ax.scatter(
            nov_pt_rpe, nov_pt_di,
            color='green', s=80, marker='*',
            edgecolor='black', zorder=5,
            label="Novelty Trial"
        )
        ax.legend(loc='best')

fig.suptitle("Δ Policy-IG vs RPE", fontsize=10, y=0.98)
fig.tight_layout(rect=[0, 0.03, 1, 0.96])
fig.savefig('costs_and_novelty.pdf', bbox_inches='tight')
plt.show()