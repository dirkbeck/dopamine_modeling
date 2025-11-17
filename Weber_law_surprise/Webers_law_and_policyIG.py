import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax, kl_div
np.random.seed(0)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.linewidth': 1.2,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.figsize': (4, 3.5),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42
})

# parameters
n_trials = 500
true_rate = 1.0
weber_w = 0.15
alpha_rate = 0.02
alpha_Q = 0.02
beta = 5.0

rate_est = true_rate
Q = np.zeros(2)
policy = softmax(beta * Q)
surprises = []
policy_IGs = []
pi_variances = []
deltas = []
trials = []

for t in range(n_trials):
    # 1) sample interval and compute surprise
    interval = np.random.exponential(1 / true_rate)
    lam1 = 1.0 / interval
    surprise = np.log2(lam1 / rate_est)

    # 2) update rate estimate
    rate_est += alpha_rate * (lam1 - rate_est)

    # 3) use surprise as delta for action 0
    delta = surprise
    Q_old = Q.copy()
    Q[0] += alpha_Q * delta

    # 4) update policy and compute change in policy-IG
    pi_old = policy.copy()
    policy = softmax(beta * Q)
    kl_value = np.sum(kl_div(policy, pi_old))

    # record
    surprises.append(surprise)
    policy_IGs.append(kl_value)
    pi_variances.append(pi_old[0] * (1 - pi_old[0]))
    deltas.append(delta)
    trials.append(t)


surprises = np.array(surprises)
policy_IGs = np.array(policy_IGs)
pi_variances = np.array(pi_variances)
deltas = np.array(deltas)
trials = np.array(trials)

early_mask = trials < 50
small_delta = np.abs(alpha_Q * deltas) < 0.1
mid_pi = (pi_variances > 0.2) & (pi_variances < 0.3)

mask = early_mask & small_delta & mid_pi

x_sim = np.abs(surprises[mask])
y_sim = policy_IGs[mask]

# ------ THEORETICAL CURVE -----
alpha = 0.02        # same as α_Q in the simulation
p0 = 0.5            # initial policy for Q = [0, 0] -> softmax(Beta*Q) = [0.5, 0.5]
Q0 = (1 / beta) * np.log(p0 / (1 - p0))  # will be zero when p0 = 0.5

delta_vals = np.linspace(0, 4, 400)

# build the exact KL for each surprise
kl_exact = []
for delta in delta_vals:
    Q_old = np.array([Q0, 0.0])
    Q_new = Q_old.copy()
    Q_new[0] += alpha * delta
    pi_old = softmax(beta * Q_old)
    pi_new = softmax(beta * Q_new)
    kl_exact.append(np.sum(kl_div(pi_new, pi_old)))
kl_exact = np.array(kl_exact)

x_theory = np.abs(delta_vals)
y_theory = kl_exact

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))

ax.plot(x_theory, y_theory, linewidth=2, color='gray', alpha=0.6, zorder=1,
        label='Theoretical curve')

ax.scatter(x_sim, y_sim, c='black', s=60, alpha=0.8, edgecolors='none', zorder=2,
          label='Simulation data')

ax.set_xlabel('|Weber law surprise|')
ax.set_ylabel('Δ policy-IG (nats)')
ax.legend(frameon=False, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)
ax.tick_params(axis='both', which='major',
               length=6, width=1.2, direction='out',
               top=False, right=False)
ax.set_xlim(0, 5.1)
ax.set_ylim(0, 0.03)
ax.set_xticks([0, 5])
ax.set_yticks([0, 0.03])
plt.tight_layout()
plt.savefig('Webers_surprise_policyIG.pdf', format='pdf')
plt.show()


plot_len_early = 20 # Number of early trials to show
x_early    = np.arange(plot_len_early)
surp_early = surprises[:plot_len_early]
ig_early   = policy_IGs[:plot_len_early]


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(4, 2.0), sharex=True)
ax0.plot(x_early, surp_early, color='blue', lw=2)
ax0.axhline(0, color='k', lw=0.8)
ax0.set_xlim(0, plot_len_early - 1)
ax0.set_xticks([0, plot_len_early])
ax0.set_ylim(surp_early.min() * 1.1, surp_early.max() * 1.1)
ax0.set_xlabel('Episode')
ax0.set_ylabel('Surprise\n(log_2 ratio)')
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.tick_params(axis='x', direction='out', length=4)

ax1.plot(x_early, ig_early, color='red', lw=2)
ax1.set_xlim(0, plot_len_early - 1)
ax1.set_ylim(0, ig_early.max() * 1.1)
ax1.set_xticks([0, plot_len_early])
ax1.set_xlabel('Episode')
ax1.set_ylabel('Δ policy-IG\n(nats)')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', direction='out', length=4)

plt.tight_layout()
plt.savefig('early_episodes_surprise_policyIG.pdf', format='pdf')
plt.show() 