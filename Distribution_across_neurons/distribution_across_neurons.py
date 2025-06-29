import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0
})

def delta_ig_nonuniform(delta, p, beta, alpha=0.5):
    dq_old = (1/beta) * np.log(p/(1-p))
    dq_new = dq_old + alpha * delta
    p_new = np.exp(beta * dq_new) / (np.exp(beta * dq_new) + 1)
    H_new = -(p_new * np.log(p_new + 1e-16) +
              (1-p_new) * np.log((1-p_new) + 1e-16))
    H_old = -(p * np.log(p) + (1-p) * np.log(1-p))
    return H_old - H_new

# parameters
beta = 3.0
alpha = 0.5
deltas = np.linspace(-2.0, 2.0, 400)

fig = plt.figure(figsize=(10, 4))

ax1 = plt.subplot(121)
ps = np.array([.6, .8, .99])
colors = plt.cm.viridis(np.linspace(0, 1, len(ps)))

for p, c in zip(ps, colors):
    DeltaIG = delta_ig_nonuniform(deltas, p, beta, alpha)
    ax1.plot(deltas, DeltaIG, color=c, linewidth=2.5, label=f"p = {p:.2f}")

ax1.axhline(0, color="black", linewidth=1, alpha=0.7)
ax1.axvline(0, color="black", linewidth=1, alpha=0.7)
ax1.set_xlabel("RPE")
ax1.set_ylabel("Delta policy-IG")
ax1.legend(loc="upper left", frameon=True)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax2 = plt.subplot(122)
all_DeltaIG = np.concatenate([delta_ig_nonuniform(deltas, p, beta, alpha) for p in ps])

counts, bins, patches = ax2.hist(all_DeltaIG, bins=30, color='#FF7F00',
                                edgecolor='black', alpha=0.8, linewidth=1.2)
bin_centers = 0.5*(bins[:-1] + bins[1:])
mu, sigma = all_DeltaIG.mean(), all_DeltaIG.std()
pdf = norm.pdf(bin_centers, loc=mu, scale=sigma)
bin_width = bins[1] - bins[0]
ax2.plot(bin_centers, pdf * all_DeltaIG.size * bin_width, 'k--',
         linewidth=2.5, label="Gaussian fit")

# compute KS
ks_stat, pval = kstest(all_DeltaIG, 'norm', args=(mu, sigma))
ax2.text(0.05, 0.95, f"KS D = {ks_stat:.3f}\np = {pval:.0e}",
         transform=ax2.transAxes, va='top', fontsize=11,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

ax2.set_xlabel("Delta Policy-IG")
ax2.set_ylabel("Count")
ax2.legend(loc="upper right")
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.subplots_adjust(wspace=0.35)


plt.tight_layout()
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
plt.savefig('distribution_across_neurons_RPE_task.pdf')
plt.show()
