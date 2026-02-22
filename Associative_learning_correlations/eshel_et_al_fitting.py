import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

df = pd.read_excel('Eshel_et_al_data_estimates.xlsx')
reward_size         = df['reward_size'].values
response_unexpected = df['response_unexpected_Fig1c'].values
response_expected   = df['response_expected_Fig2a'].values

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def H_bin(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def policy_ig(r, k, beta, gamma, offset):
    p = sigmoid(beta * np.power(np.maximum(r, 1e-10), gamma))
    return k * (np.log(2) - H_bin(p)) + offset

def policy_ig_expected(r, k, beta, gamma, offset, p_cue):
    p_r = sigmoid(beta * np.power(np.maximum(r, 1e-10), gamma))
    return k * (H_bin(p_cue) - H_bin(p_r)) + offset

def r2(y, yhat):
    return 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)

popt, _ = curve_fit(policy_ig, reward_size, response_unexpected,
                    p0=[16, 0.5, 0.5, 0.5],
                    bounds=([1, 0.01, 0.05, -5], [100, 20, 2.0, 5]),
                    maxfev=50000)
k, beta, gamma, offset = popt

task_rewards = np.array([0.1, 0.3, 1.2, 2.5, 5, 10, 20])
p_cue = sigmoid(beta * np.mean(np.power(task_rewards, gamma)))

r2_unexp = r2(response_unexpected, policy_ig(reward_size, *popt))
r2_exp = r2(response_expected, policy_ig_expected(reward_size, *popt, p_cue))

print(f"Unexpected R² = {r2_unexp:.4f}")
print(f"Expected   R² = {r2_exp:.4f}")
print(f"k={k:.2f}  β={beta:.3f}  γ={gamma:.3f}  offset={offset:.2f}")

r_fine = np.linspace(0.05, 22, 500)
FONT = 14

fig, ax = plt.subplots(figsize=(2.6, 2.0))

ax.plot(r_fine, policy_ig(r_fine, *popt), color='#E07030', lw=1.6,
        label=f'unexpected (R²={r2_unexp:.3f})')
ax.plot(r_fine, policy_ig_expected(r_fine, *popt, p_cue), color='#333333', lw=1.6,
        label=f'expected (R²={r2_exp:.3f})')
ax.scatter(reward_size, response_unexpected, s=28, c='#E07030', edgecolors='k', linewidth=0.5, zorder=5)
ax.scatter(reward_size, response_expected,   s=28, c='#333333', edgecolors='k', linewidth=0.5, zorder=5)

ax.axhline(0, color='#AAAAAA', lw=0.5)
ax.set_xticks([0, 10, 20])
ax.set_yticks([0, 5, 10])
ax.set_xlabel('Reward size (μL)', fontsize=FONT)
ax.set_ylabel('Firing rate\n(sp/s above baseline)', fontsize=FONT)
ax.tick_params(labelsize=FONT)
ax.set_xlim(-0.5, 22)
ax.set_ylim(-3, 14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(fontsize=9, frameon=False, loc='upper left')

plt.tight_layout()
plt.savefig('eshel_policyIG_fit.pdf', dpi=300)
plt.show()