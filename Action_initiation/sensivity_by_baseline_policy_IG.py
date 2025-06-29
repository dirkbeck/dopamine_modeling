import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import matplotlib as mpl

# fixed vals
Q1 = 1.0
Q2 = 0.0
ΔQ = Q1 - Q2

def policy_p1(beta):
    z = np.clip(beta * ΔQ, -700, +700)
    return 1.0 / (1.0 + np.exp(-z))

def entropy(p1):
    p1 = np.clip(p1, 1e-10, 1-1e-10)
    p2 = 1 - p1
    return -(p1*np.log(p1) + p2*np.log(p2))

MAX_ENTROPY = np.log(2.0)

def find_beta(H_target):
    if abs(entropy(policy_p1(0)) - H_target) < 1e-6:
        return 0.0
    def f(beta): return entropy(policy_p1(beta)) - H_target
    try:
        β_root, r = brentq(f, 0.0, 50.0, full_output=True, xtol=1e-6)
        return β_root if r.converged else np.nan
    except ValueError:
        return np.nan

# sweep entropies
Hs = np.linspace(0.01, MAX_ENTROPY - 0.01, 100)
betas, p1s, policy_IG = [], [], []
for H in Hs:
    β = find_beta(H)
    if not np.isnan(β):
        p = policy_p1(β)
        betas.append(β)
        p1s.append(p)
        policy_IG.append(MAX_ENTROPY - H)

betas = np.array(betas)
p1s = np.array(p1s)
policy_IG = np.array(policy_IG)

# sensitivity
sens = np.abs(np.gradient(p1s, policy_IG))

# plot
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.8,
    'figure.figsize': (2,2),
    'figure.dpi': 300,
    'pdf.fonttype': 42,
})
fig, ax = plt.subplots()
ax.plot(policy_IG, sens, 'k-', lw=1.8)
ax.set_xlabel('Policy-IG (nats)', labelpad=2)
ax.set_ylabel('Sensitivity (nat$^{-1}$)', labelpad=2)
ax.set_title('Policy-IG Sensitivity', pad=4)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(policy_IG.min(), policy_IG.max())
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('sensitivity_vs_policyIG.pdf', bbox_inches='tight')
plt.show()