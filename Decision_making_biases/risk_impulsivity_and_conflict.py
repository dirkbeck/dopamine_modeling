import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})
mpl.rcParams['pdf.fonttype'] = 42

def calculate_pi_u(beta, p, Delta):
    arg = np.clip(beta * Delta, -700, 700)
    return p * (1.0 / (1.0 + np.exp(arg)))

def calculate_pi_LL_with_discounting(beta, A_ss, A_ll, D, k):
    Q_ss = A_ss
    Q_ll = A_ll / (1.0 + k * D)
    arg = np.clip(beta * (Q_ss - Q_ll), -700, 700)
    return 1.0 / (1.0 + np.exp(arg))

def calculate_pi_hchr(beta, engage, Delta):
    arg = np.clip(beta * Delta, -700, 700)
    return engage * (1.0 / (1.0 + np.exp(arg)))

# risk taking
beta_low, beta_high = 0.2, 5.0
p_vals = np.linspace(0.01, 1.0, 100)
y_vals = np.linspace(0, 3, 100)
P, Y = np.meshgrid(p_vals, y_vals)
Delta = -Y

pi_low = calculate_pi_u(beta_low, P, Delta)
pi_high = calculate_pi_u(beta_high, P, Delta)

idx = np.linspace(0, len(p_vals)-1, 5, dtype=int)
p_sel = p_vals[idx]

fig, axes = plt.subplots(1,2, figsize=(6,3), sharey=True, dpi=300)
cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=p_vals.min(), vmax=p_vals.max())

ax = axes[0]
for i, p_idx in enumerate(idx):
    c = cmap(norm(p_sel[i]))
    ax.plot(y_vals, pi_low[:, p_idx], color=c, linewidth=3, alpha=0.9)
ax.set_title('Lower policy-IG')
ax.set_xlabel('Value difference\n(uncertain − certain)')
ax.set_ylabel('Choice probability')
ax.set_ylim(0, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 3.2)

ax = axes[1]
for i, p_idx in enumerate(idx):
    c = cmap(norm(p_sel[i]))
    ax.plot(y_vals, pi_high[:, p_idx], color=c, linewidth=3, alpha=0.9)
ax.set_title('Higher policy-IG')
ax.set_xlabel('Value difference\n(uncertain − certain)')
ax.set_ylim(0, 1.05)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 3.2)

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, aspect=15, pad=0.05)
cbar.set_label('Uncertain action\navailability')

mid = len(y_vals)//2
for i, p_idx in enumerate(idx):
    c = cmap(norm(p_sel[i]))
    axes[1].text(3.05, pi_high[mid, p_idx], f'p={p_sel[i]:.2f}',
                 color=c, ha='left', va='center')

plt.tight_layout()
fig.subplots_adjust(top=0.88, right=0.85)
fig.savefig('policy_psychometric_functions.pdf', bbox_inches='tight')
plt.show()

# delay discounting
A_LL, k = 10.0, 0.1
beta_low, beta_high = 0.5, 2.0
A_SS = np.linspace(0, 15, 100)
x_axis = A_LL - A_SS
delays = np.array([0,5,10,20,40,60])

fig, axes = plt.subplots(1,2, figsize=(6,3), sharey=True, dpi=300)
cmap = plt.get_cmap('viridis', len(delays))
norm = mpl.colors.Normalize(vmin=delays.min(), vmax=delays.max())

for ax, beta, title in zip(axes, (beta_low,beta_high), ('Lower policy-IG','Higher policy-IG')):
    ax.set_title(title)
    for i, D in enumerate(delays):
        pi = calculate_pi_LL_with_discounting(beta, A_SS, A_LL, D, k)
        ax.plot(x_axis, pi, color=cmap(i), linewidth=2.5)
    ax.set_xlabel('Value difference\n(later − sooner)')
    ax.set_ylim(-0.05,1.05)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Larger/later probability')

sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array(delays)
cbar = fig.colorbar(sm, ax=axes.ravel(), shrink=0.9, aspect=15, pad=0.05)
cbar.set_label('Delay time for LL')

plt.tight_layout(rect=[0,0,0.93,1.0])
fig.savefig('delay_discounting.pdf')
plt.show()

# conflict
costs = np.linspace(0,5,2)
rewards = np.linspace(0,10,2)
C, R = np.meshgrid(costs, rewards)

R_LCLR, C_LCLR = 2.0, 1.0
Q_LCLR = R_LCLR - C_LCLR
DeltaQ = Q_LCLR - (R - C)

beta_low, beta_high = 0.2, 5.0
engage = 1.0

pi_low = calculate_pi_hchr(beta_low, engage, DeltaQ)
pi_high = calculate_pi_hchr(beta_high, engage, DeltaQ)

fig, axes = plt.subplots(1,2, figsize=(6,3), dpi=300)
for ax, data, title in zip(axes, (pi_low,pi_high), ('Lower policy-IG','Higher policy-IG')):
    im = ax.pcolormesh(C, R, data, cmap='bwr', vmin=0, vmax=1, shading='nearest')
    ax.set_title(title)
    ax.set_xlabel('Cost of HCHR')
    if ax is axes[0]:
        ax.set_ylabel('Reward of HCHR')
    ax.axis('tight')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.88,0.15,0.03,0.7])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Prob(HCHR)')
plt.tight_layout()
fig.savefig('conflict_task_heatmap.pdf')
plt.show()
